# Identity Guard

This repository contains the software for a thesis on detecting the unauthorized use of a person's facial images online. Identity Guard is a web application: the user uploads a single photo of their face, the system searches publicly indexed sources for images that contain the same person, confirms each candidate with face verification, and passes the confirmed matches to a deepfake detector that indicates whether the face may have been synthetically altered (face swap). The pages of the confirmed matches are then scanned for other images of the same person, and every deepfake found is searched again to trace the other sites that posted it. The result is a report listing where the face appears, how close each match is, and which matches look manipulated, together with per stage statistics and the total analysis time.

## Motivation

Photos posted on social media and websites can be copied, reposted, or reused in a misleading context without the consent of the person shown. At the same time, deep learning makes it easy to produce realistic manipulated images, most notably face swaps, where the face of one person is placed onto a real photograph of another. A person who wants to know where their face appears online therefore faces two questions: where is this photo (or my face) being used, and has any of those images been altered?

Existing tools answer only parts of this. Reverse image search engines find visually similar pictures, but return many images of other people and give no indication of manipulation. Face recognition libraries can tell whether two faces belong to the same person, but do not search the web. Deepfake detectors estimate whether an image is synthetic, but are known to generalize poorly to manipulation types, resolutions, and compression levels they were not trained on. This project combines all three into a single pipeline:

1. **Reverse image search** (Google Lens through SerpApi) finds candidate images.
2. **Face verification** (DeepFace with Facenet512 and MTCNN) keeps only candidates that contain the same person.
3. **Deepfake detection** (UniversalFakeDetect served by DeepSafe) flags confirmed matches that look manipulated.
4. **Page scan and deepfake tracing** opens the pages of the confirmed matches to find other images of the person posted there, and runs a new reverse search for every deepfake found, to list the other sites that host it.

The contribution is not a new recognition or detection algorithm, but the integration of existing ones into a working prototype, and an experimental study of where the off-the-shelf components fail and how to fix them. The main findings were:

- With the initial configuration (VGG-Face at a 0.45 distance threshold), **none** of the author's face swap deepfakes passed the face filter, so they never reached the detector. Switching to Facenet512 at 0.60 fixed this.
- The NPR detector returned 0.0 on every image and was removed. UniversalFakeDetect center crops its input to 224x224, so on large phone photos it often never saw the face; sending only the cropped face fixed this.
- UniversalFakeDetect was not trained on face swaps. Its final linear layer was retrained on DeepFakeFace face swaps and on real faces from DeepFakeFace and FFHQ. On 1,058 unseen test images this raised accuracy from **81.3% to 92.9%**, with **95.3%** of face swaps caught and **8.5%** of real photos flagged, and 14 of the author's 15 test images were classified correctly (all 11 real photos and 3 of 4 deepfakes).
- Reverse search returns visually similar pictures, so a deepfake of a person (a different picture with their face) is not returned by a search with their real photo. Scanning the pages of confirmed matches finds such deepfakes when they are posted next to a real photo of the person. A controlled test showed that deepfakes posted on a page with no real photo of the person are not reachable through public reverse search, even when the query photo's face is very close to them; this is a limit of the search engines, not of the pipeline.

The deepfake indication is presented as supporting evidence, never as proof, and the system is designed for locating reposts of a specific photo rather than mapping a person's entire online presence.

## Repository Structure

```
Identity-Guard/
├── app/
│   ├── config.py               # All thresholds and settings in one place (keys read from .env)
│   ├── reverse_search.py       # Module 1: SerpApi upload, Google Lens / Yandex search, image download
│   ├── face_matcher.py         # Module 2: MTCNN detection, Facenet512 embeddings, distance, face crop
│   ├── deepfake_client.py      # Module 3: client for the DeepSafe /predict API
│   ├── page_scan.py            # Module 4: collects the images of the pages where matches were found
│   ├── pipeline.py             # Module 5: runs the stages in order, keeps statistics, builds the report
│   ├── main.py                 # FastAPI server: web page, /api/status, /api/analyse
│   └── static/index.html       # Single page user interface (HTML, CSS, JavaScript)
├── models/
│   └── fc_weights_faceswap.pth # Retrained final layer for UniversalFakeDetect
├── retraining/                 # Scripts and training log that produced the weights above
│   ├── make_crops.py           # Face crops from DeepFakeFace (split by identity) and own photos
│   ├── make_crops2.py          # Adds FFHQ real faces and degraded training copies
│   ├── extract_feats.py        # CLIP ViT-L/14 features, runs inside the DeepSafe container
│   ├── train_fc.py             # Logistic regression, picks L2 strength and threshold on validation
│   ├── install_fc.py           # Installs the new weights into the container
│   ├── verify_api.py           # Checks the installed weights through the real /predict endpoint
│   └── train_fc_log.txt        # Output of the run that produced models/fc_weights_faceswap.pth
├── compare_engines.py          # Google Lens vs Yandex contribution for one publicly hosted image
├── measure_search.py           # Search + face matching measurements over a folder (no DeepSafe)
├── label_matches.html          # Local page to label measure_search hits and compute precision/recall
├── deepsafe_check.py           # Standalone DeepSafe test, optionally on degraded copies (--degrade)
├── sanity_check.py             # Sends one image straight to DeepSafe
├── start-identity-guard.bat    # Windows launcher for DeepSafe and the app
├── requirements.txt            # Pinned Python dependencies
└── .env.example                # Template for SERPAPI_KEY, DEEPSAFE_URL, DEEPSAFE_ENSEMBLE
```

## How It Works

The system is a sequential pipeline. Each stage takes the output of the previous one and narrows down the set of images that move on, so that the expensive stages run only where they are needed, and a failure early on stops the analysis before any wasted work. The flow is run by `pipeline.run()`:

```
uploaded photo
  -> 1. reference face embedding (Facenet512)        0 or >1 faces: HTTP 400, nothing searched
  -> 2. reverse image search (Google Lens, SerpApi)
  -> 3. download candidate images
  -> 4. face filter (cosine distance < 0.60)         other people are discarded
  -> 5. page scan of the confirmed matches' pages     -> face filter again (free, no searches)
  -> 6. deepfake check on the cropped face (DeepSafe, threshold 0.24)
  -> 7. reverse search with each deepfake found       -> copies of that deepfake (2 searches each)
  -> report (JSON, rendered by the web page)
```

1. **Reference face.** The uploaded photo (JPEG, PNG or WebP, up to 10 MB) is resized to at most 1024 px and must contain exactly one face. DeepFace detects it with MTCNN and computes a 512 dimensional Facenet512 embedding. If no face or several faces are found, `NoFaceFoundError` or `MultipleFacesError` is raised and the server returns HTTP 400 before any search quota is spent, since it is not clear whom to look for.
2. **Reverse image search.** SerpApi accepts uploads up to about 500 KB, so larger photos are recompressed to JPEG at decreasing quality (85, 70, 55, 40, 25) and, if still too large, shrunk. The image is uploaded once, and the returned `image_id` is used for two Google Lens calls in parallel (`asyncio.gather`): visually similar images and exact matches. When a public URL of the photo is supplied (as `compare_engines.py` does), Yandex is searched as well, since SerpApi's Yandex engine accepts only public URLs. For every result, the image URL is taken from the original image if available, then the `image` field, then the thumbnail, and each hit is marked as original or thumbnail. Results are merged and deduplicated by image URL.
3. **Download.** Candidates are downloaded concurrently (at most 5 at a time, 30 s timeout) with a browser User-Agent header, because many servers reject requests without one. Non-image responses and failed downloads (for example HTTP 502) are recorded on the hit and skipped without stopping the analysis. Images are kept in memory only.
4. **Face filter.** Each downloaded image is embedded the same way, and the cosine distance to the reference is computed for every face in it; the closest face counts. Candidates under `FACE_DISTANCE_THRESHOLD` (0.60) are confirmed as the same person. This stage exists because reverse search returns visually similar images, not necessarily the same person.
5. **Page scan.** Reverse search finds pictures that look like the query, so a deepfake of the user (a different picture with their face on another body and background) is not returned. When a page that hosts a confirmed match also hosts deepfakes, they can still be found there. `page_scan.scan_pages()` opens the pages of the confirmed matches (at most 5), collects their images from `img` tags, lazy loading attributes, the largest `srcset` entry, `og:image` / `twitter:image` and links that point straight to image files (at most 30 per page, SVG, ICO and GIF skipped), downloads them, skips those smaller than 100 px, and passes the rest through the same face filter. This uses no SerpApi searches. The result is other photos and deepfakes of the user on the same sites.
6. **Deepfake check.** For every confirmed match, the matching face is cropped as a square with a 30% margin and resized to 256x256, and only this crop is sent to DeepSafe. UniversalFakeDetect returns a probability of manipulation; values at or above `DEEPSAFE_THRESHOLD` (0.24) give the verdict "fake". Requests run sequentially because the models run locally on CPU. Compressed Yandex thumbnails are skipped because they produced false alarms on real photos.
7. **Deepfake tracing.** For each confirmed match flagged as fake (at most 3), the deepfake itself is used as a new reverse search query, to find the other sites that posted it. Results are compared with the face of the deepfake (not with the user's photo, since a swap can be some distance from it), so the filter keeps copies of the same deepfake; they then go through the deepfake check and are marked with `traced_from`. Each traced deepfake costs 2 SerpApi searches.
8. **Report.** Confirmed findings are sorted by increasing distance and returned with their source page, title, engine, match type, image kind, distance, similarity (100 x (1 - distance), not a probability), verdict and model probability, plus the per stage counters and total duration.

The pipeline keeps counters for each stage, which show where images are found and lost:

| Counter | Meaning |
|---|---|
| `found_by_search` | Unique results returned by the search engines |
| `downloaded` | Candidate images that were downloaded successfully |
| `contain_a_face` | Downloaded images in which a face was detected |
| `page_images` | Images downloaded from the pages of the confirmed matches |
| `traced_deepfakes` | Deepfakes used as a new search query |
| `trace_results` | Images downloaded from the deepfake searches |
| `face_matches` | Images confirmed as the same person (from search, page scan and tracing) |
| `flagged_synthetic` | Confirmed matches flagged as likely manipulated |

## Module Reference

### `app/config.py`
Single place for every tunable setting, so experiments change one file and not the module code. The SerpApi key, DeepSafe URL and ensemble method are read from `.env` with python-dotenv.

| Setting | Value | Purpose |
|---|---|---|
| `SEARCH_ENGINES` | `["google_lens"]` | Engines used by the web app (Yandex needs a public URL) |
| `MAX_RESULTS_PER_ENGINE` | 40 | Results kept per engine |
| `FACE_DISTANCE_THRESHOLD` | 0.60 | Facenet512 cosine distance below which two faces are the same person |
| `MAX_IMAGE_DIMENSION` | 1024 | Longest side before embedding, for speed |
| `FACE_CROP_MARGIN`, `FACE_CROP_SIZE` | 0.3, 256 | Margin and size of the face crop sent to DeepSafe |
| `DEEPSAFE_MODELS` | `["universalfakedetect"]` | Detectors called through DeepSafe |
| `DEEPSAFE_ENSEMBLE` | `average` | With one model, returns that model's probability directly |
| `DEEPSAFE_THRESHOLD` | 0.24 | Fake threshold, valid only with `models/fc_weights_faceswap.pth` |
| `PAGE_SCAN_ENABLED` | True | Scan the pages of confirmed matches |
| `PAGE_SCAN_MAX_PAGES`, `PAGE_SCAN_MAX_IMAGES` | 5, 30 | Pages scanned per analysis and images kept per page |
| `PAGE_SCAN_MIN_SIDE` | 100 px | Smaller page images (icons, logos) are skipped |
| `DEEPFAKE_TRACE_MAX` | 3 | Deepfakes searched again per analysis (2 searches each) |
| `HTTP_TIMEOUT` | 240 s | Timeout for external API calls |
| `DOWNLOAD_CONCURRENCY` | 5 | Parallel candidate downloads |

### `app/reverse_search.py`
Implements the search stage. `SearchHit` is a dataclass carrying the image URL, source page, title, engine, match type and image kind, plus fields filled in by later stages (face distance, match flag, deepfake verdict and confidence, model results, error, and `traced_from`, the deepfake whose search found the hit). `upload_image()` uploads the query to SerpApi, using `_ensure_under_limit()` to recompress it below the size limit. `_search_one_engine()` and `_search_exact_matches()` run the two Google Lens calls, `_search_yandex_by_url()` the optional Yandex call, and `_parse_google_lens()` / `_parse_yandex()` turn each response into `SearchHit` records. `search()` uploads once, runs all searches in parallel and deduplicates; `download_images()` fetches the candidates with limited concurrency and records failures per hit.

### `app/face_matcher.py`
Wraps DeepFace with `MODEL_NAME = "Facenet512"` and `DETECTOR_BACKEND = "mtcnn"`. `_load_and_resize()` converts to RGB and limits the size, `_save_temp()` writes a temporary file because DeepFace works most reliably with file paths, and `_extract_embedding()` returns the embeddings of all detected faces (empty list if none). `encode_query_face()` enforces exactly one face on the uploaded photo. `_cosine_distance()` computes 1 - cos(A, B), and `compare_with_candidate()` returns the match decision and the smallest distance over the candidate's faces. `crop_face()` finds the face closest to the reference and crops it for the detector, and `distance_to_similarity()` converts a distance into the percentage shown in the report.

### `app/deepfake_client.py`
Client for DeepSafe. `check_health()` tells `/api/status` whether the service is up. `analyse_one()` base64 encodes an image and POSTs it to `/predict` with the configured models, ensemble method and threshold. `analyse_batch()` analyses images one after another with a short pause, storing either the result or the error for each image so one failure does not affect the others.

### `app/page_scan.py`
Implements the page scan. `_ImageCollector` is an `html.parser` based parser that gathers image URLs and the page title. `_scan_one()` downloads one page with a browser User-Agent and returns its images as `SearchHit` records with engine `page_scan` and match type `same_page`. `scan_pages()` scans the source pages of the confirmed matches in parallel and drops images already seen in the search. Pages that build their content with JavaScript only, or that require a login, return no images.

### `app/pipeline.py`
Orchestrates the modules. The async `run()` executes the stages above and updates the counters. `_filter_faces()` applies the face filter to a list of hits against a given face embedding (the user's for search and page scan results, the deepfake's for traced copies). `_check_deepfakes()` crops the faces, calls DeepSafe and fills in the verdicts, skipping Yandex thumbnails. `_trace_deepfake()` searches with a deepfake and keeps its copies. `_build_report()` converts confirmed hits to dictionaries, adds the similarity percentage, sorts them by distance and returns the statistics, findings and duration ready for JSON serialization.

### `app/main.py`
FastAPI application (served by uvicorn on port 8080, since DeepSafe uses 8000). `/` serves the interface; `/api/status` reports whether the SerpApi key is configured, whether DeepSafe is reachable, the current face threshold and the engines; `/api/analyse` validates the upload (type and size) and runs the pipeline. Module errors become HTTP 400 (no face or several faces, bad file), 503 (missing configuration) or 500 (unexpected, logged).

### `app/static/index.html`
Single page interface in Greek, written in plain HTML, CSS and JavaScript without a framework. On load it checks `/api/status` and warns if a service is missing. The user drops or selects a photo, sees a preview, and starts the analysis; four progress steps are shown on timers, since the server does not stream real progress. The report shows four counters (search results, same face, page images, likely altered) and the findings in up to three sections: from the reverse search, from the same pages ("Στις ίδιες σελίδες"), and the other sites where a deepfake was found ("Το deepfake αναρτήθηκε και εδώ"). Each finding is a card with source, title, similarity, distance and an alteration label: "Πιθανή αλλοίωση" for fake, "Δεν εντοπίστηκε αλλοίωση" for real, and "Χωρίς έλεγχο αλλοίωσης" when no check was made. Findings from thumbnails are marked as less reliable.

### `retraining/`
UniversalFakeDetect is a frozen CLIP ViT-L/14 image encoder followed by a single linear layer (768 inputs, 1 output). Only that layer is retrained, so the improvement needs no change to DeepSafe or to the application code: only the weight file and the threshold change.

- `make_crops.py` crops faces with exactly the app's geometry (30% margin, 256 px PNG). DeepFakeFace is split by identity folder (00 to 69 train, 70 to 84 validation, 85 to 99 test) so that no person appears in two splits. The author's own photos form a separate `own` split used only for the final check.
- `make_crops2.py` adds FFHQ real faces (first 800 train, next 200 validation, rest test) and one degraded copy of half the training crops (rescale to 35% to 100%, JPEG quality 35 to 90, occasional light blur), so the model does not associate image quality with manipulation.
- `extract_feats.py` runs inside the DeepSafe container with the same transform as the detector (center crop 224 and CLIP normalization), saves the 768 dimensional features and the original layer, and can resume from a partial file.
- `train_fc.py` fits an L2 regularized logistic regression in NumPy. The L2 strength and the threshold are chosen on the validation split only (most face swaps caught with at most 8% of real photos flagged); the test split and the author's photos are scored once at the end, next to the original layer.
- `install_fc.py` converts the result into a PyTorch state dict, keeps a copy of the original weights, and installs the new file where the detector loads it.
- `verify_api.py` sends the author's photos and 100 random test crops through the real `/predict` endpoint and confirms the API probabilities match the offline ones.

### Experiment scripts
These are not part of the application and exercise one part of the pipeline at a time.

- `sanity_check.py` sends one image directly to DeepSafe, bypassing search and face matching.
- `deepsafe_check.py` analyses files or folders; with `--degrade` it also sends resized (512, 256, 128 px), JPEG compressed (quality 90, 70, 50) and thumbnail-like (150 px, quality 75) versions and writes all model probabilities to CSV.
- `compare_engines.py` runs the full pipeline for a local image and its public URL, so that Yandex is searched too, and prints the confirmed findings per engine.
- `measure_search.py` runs search, download and face distance for every result (not only matches) over a folder of images and repeated runs, and writes per hit and per run CSV files, including recall against the other experiment images. With `--crop` it searches with the cropped face instead of the whole photo.
- `label_matches.html` is a local page for labelling the `<out>_hits.csv` of `measure_search.py`. It shows every hit with a face up to a maximum distance (0.70), sorted by distance with a dashed line at the system threshold, and each one is labelled as same person, not, or unclear. It computes TP, FP, FN, TN, precision and recall with Wilson 95% intervals for thresholds 0.35 to 0.50, keeps the labels in the browser and exports them as `labels.csv`.

## Experimental Results

All figures below come from the thesis evaluation. Detection and face verification were measured on the author's 11 real photos and 4 face swaps (created with Magic Hour), a StyleGAN image used as a positive control, 64 photos of unrelated people, and the public DeepFakeFace (1,200 real and 1,200 InsightFace face swaps) and FFHQ datasets.

### Face verification model and threshold

Share of pairs that pass the face filter (own photo pairs / own photo vs deepfake pairs / pairs with strangers, 110 / 44 / 704 pairs):

| Threshold | VGG-Face | Facenet512 | ArcFace |
|---|---|---|---|
| 0.45 | 18.2% / 9.1% / 0.0% | 70.9% / 50.0% / 0.1% | 16.4% / 9.1% / 0.0% |
| 0.50 | 25.5% / 15.9% / 0.0% | 76.4% / 63.6% / 0.4% | 29.1% / 29.5% / 0.0% |
| 0.55 | 34.5% / 25.0% / 0.0% | 83.6% / 81.8% / 0.6% | 38.2% / 45.5% / 0.0% |
| **0.60** | 40.0% / 27.3% / 0.0% | **92.7% / 90.9% / 1.6%** | 61.8% / 65.9% / 0.0% |
| 0.65 | 52.7% / 34.1% / 0.1% | 94.5% / 97.7% / 3.6% | 67.3% / 77.3% / 0.0% |

Facenet512 at 0.60 keeps over 90% of the user's photos and deepfakes while admitting 1.6% of strangers, and was adopted. With VGG-Face at 0.45, no deepfake reached the detector.

### Detector input

UniversalFakeDetect on the author's 15 images at threshold 0.5 (deepfakes caught / real photos flagged):

| Input to the detector | Caught | Flagged |
|---|---|---|
| Whole image, original resolution | 2/4 | 3/11 |
| Whole image, 512 px | 3/4 | 0/11 |
| **Face crop, 30% margin, 256 px** | **4/4** | **0/11** |
| Face crop, 30% margin, 512 px | 2/4 | 0/11 |
| Face crop, 100% margin, 256 px | 2/4 | 0/11 |

On 400 DeepFakeFace images the same setup reached only AUC 0.875 and 76.2% accuracy at 0.5 (54.5% of face swaps caught), showing that the limit was the model itself, which had not been trained on face swaps.

### Retraining the final layer

| Setup | Test images | AUC | Accuracy | Face swaps caught | Real flagged | Author's real photos flagged |
|---|---|---|---|---|---|---|
| Original layer, threshold 0.06 | 1,058 | 0.870 | 81.3% | 75.3% | 15.3% | 2/11 |
| Round 1: DeepFakeFace only, threshold 0.40 | 762 | 0.988 | 93.3% | 93.5% | 6.9% | 4/11 |
| **Round 2: + FFHQ + degraded copies, threshold 0.24** | 1,058 | **0.987** | **92.9%** | **95.3%** | **8.5%** | **0/11** |

Round 1 flagged 4 of the author's real photos because DeepFakeFace's real images are mostly old Wikipedia photos, so a clean modern phone photo looked more like a face swap (domain gap). Adding modern FFHQ faces and degraded copies fixed this. With the final weights, none of the 296 FFHQ test photos was flagged; the remaining false alarms are concentrated in old, blurry, scanned or very small Wikipedia photos (15.1%). The author's real photos scored at most 0.04. One deepfake, `df_3`, contains a second, blurred face in the background: the retraining evaluation cropped that face (0.35, counted as caught), while the application crops the face that matches the user, which is the swapped face and scores 0.05, so it is missed. Under the application's conditions the result on the author's images is 14/15 (3 of 4 deepfakes caught, no real photo flagged). The full output is in `retraining/train_fc_log.txt`.

### Initial vs final configuration

| Metric | Initial | Final |
|---|---|---|
| Deepfakes passing the face filter | 9.1% | 90.9% |
| Own photos passing the face filter | 18.2% | 92.7% |
| Strangers passing the face filter | 0.0% | 1.6% |
| Detector accuracy, test set (1,058) | 81.3% | 92.9% |
| Face swaps caught, test set | 75.3% | 95.3% |
| Real photos flagged, test set | 15.3% | 8.5% |
| Correct decisions on the author's 15 images | 12/15 | 14/15 |

### End to end run and timing

A full analysis of the face swap `df_1` with the final system returned 41 unique search results, 27 downloaded (all with a face), 8 confirmed as the same face and 4 flagged as likely altered, in 67.3 s. Both copies of `df_1` itself (on GitHub Pages and as a SerpApi thumbnail) were correctly flagged; the other six matches were photos of the public figure whose image was likely the base of the swap.

| Stage | Duration |
|---|---|
| Search (upload + 2 Lens calls) | about 5 s |
| Download of 41 candidates | about 4 s |
| Face verification (27 images) | about 24 s (0.9 s per image) |
| Deepfake check (8 faces) | about 28 s |
| **Total** | **67.3 s** |

### Search recall

Google Lens reliably finds copies of the same photo but rarely other photos of the same person: across 10 queries, only 1 of 140 possible matches to other experiment photos was retrieved (0.7%, a lower bound since thumbnails do not carry the file name). Between 20% and 35% of candidate images fail to download.

### Page scan and deepfakes on isolated pages

Starting from a real photo, the page scan finds the deepfakes posted on the same page. On the test page, one confirmed image led to all other images on the page, and 3 of the 4 deepfakes were flagged (`df_3` is the one discussed above). Tracing each deepfake then lists the other sites that host it.

To check whether deepfakes can be found when no real photo of the person is on the same page, the real photos were removed from the test site, leaving only the deepfakes, and two new photos of the author that had never been published were used as queries. Google had indexed the deepfakes: searching with `df_1` returned `df_1` itself.

| Query | Results | Same face (d < 0.60) | Deepfakes returned |
|---|---|---|---|
| `new.jpg` (glasses), whole photo | 40 | 11, all other people | 0 |
| `new.jpg` (glasses), face crop | 40 | 11, all other people | 0 |
| `new2.jpg`, whole photo | 40 | 7, all other people | 0 |
| `new2.jpg`, face crop | 40 | 9, all other people | 0 |

Even with `new2.jpg`, whose face distance to the four deepfakes is 0.25 to 0.39, no search returned a deepfake: Google Lens ranks by overall appearance (hair, pose, lighting) and returns lookalikes, not the same identity. Searching with the face crop changed which lookalikes were returned but not the outcome. Deepfakes that are not linked to any real photo of the person are therefore out of reach of public reverse search; reaching them would need search by facial identity (as in services such as PimEyes or FaceCheck.ID) or prevention by image hashing (such as StopNCII). Two side observations: with glasses on the query photo, 3 of the 4 deepfakes were above the 0.60 distance threshold, so a reference photo without glasses is recommended, and Google kept returning images for some time after they were deleted from the site.

## Running the Application

Requirements: Python 3.11 (TensorFlow and DeepFace did not support 3.14 during development), Docker, and a SerpApi key (the free plan allows 250 searches per month; each analysis uses two).

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your SERPAPI_KEY
```

Start DeepSafe in a separate folder. The current DeepSafe image does not install OpenCV for UniversalFakeDetect, which makes the container restart in a loop (`No module named 'cv2'`). Before building, add `opencv-python-headless==4.11.0.86` to `models/image/universalfakedetect/requirements.txt`:

```bash
git clone https://github.com/siddharthksah/DeepSafe.git
cd DeepSafe
docker compose up -d --build api universalfakedetect
```

Then, from this repository, install the retrained weights and start the app:

```bash
docker cp models/fc_weights_faceswap.pth deepsafe-universalfakedetect:/app/model/universalfakedetect/pretrained_weights/fc_weights.pth
docker restart deepsafe-universalfakedetect
uvicorn app.main:app --port 8080
```

Open http://localhost:8080. DeepSafe listens on port 8000 (http://localhost:8000/docs). The first call to UniversalFakeDetect is slow (up to 3 to 4 minutes) while CLIP loads. The weights must be copied again whenever the container is rebuilt, and the 0.24 threshold is only valid with these weights.

## Reproducing the Retrained Detector

The datasets are not included because of their size and licenses:

- [DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace): `wiki.zip` (real) and `insight.zip` (face swaps), extracted to `dff/wiki` and `dff/insight`.
- [FFHQ 512](https://huggingface.co/datasets/Ryan-sjtu/ffhq512-caption): the first parquet file, used as additional real faces.
- Optionally a folder of your own photos named `me_*` (real) and `df_*` (face swaps), used only for the final check.

With DeepSafe running, from this repository:

```bash
python retraining/make_crops.py crops dff own_images
python retraining/make_crops2.py crops ffhq.parquet
docker exec deepsafe-universalfakedetect mkdir -p /work
docker cp crops deepsafe-universalfakedetect:/work/crops
docker cp retraining/extract_feats.py deepsafe-universalfakedetect:/work/
docker exec deepsafe-universalfakedetect python /work/extract_feats.py /work/crops /work/feats.npz
docker cp deepsafe-universalfakedetect:/work/feats.npz .
python retraining/train_fc.py feats.npz fc_faceswap.npz
docker cp fc_faceswap.npz deepsafe-universalfakedetect:/work/
docker cp retraining/install_fc.py deepsafe-universalfakedetect:/work/
docker exec deepsafe-universalfakedetect python /work/install_fc.py
docker restart deepsafe-universalfakedetect
python retraining/verify_api.py crops feats.npz fc_faceswap.npz 0.24
```

Feature extraction takes about 0.7 s per crop on a CPU (roughly 1 to 2 hours for the 5,300 crops); training the layer takes seconds, since features are computed once.

| Split | DeepFakeFace folders | FFHQ images | Total crops |
|---|---|---|---|
| Train | 00 to 69 | 800 | 3,472 (including degraded copies) |
| Validation | 70 to 84 | 200 | 759 |
| Test | 85 to 99 | 296 | 1,058 |

## Limitations

- **External services.** Google Lens has no official API, so searches go through SerpApi, which adds a dependency, a cost per search, and the upload of the user's photo to a third party (a GDPR consideration).
- **Search coverage.** Only publicly indexed content is reachable; private profiles are not. Engines return a limited number of results, often thumbnails, and recall of other photos of the same person is low.
- **Yandex.** SerpApi's Yandex engine accepts only public URLs; temporary tunnels (ngrok, Cloudflare Tunnel) returned no results, so Yandex is used only in experiments with images hosted on GitHub Pages.
- **Isolated deepfakes.** Deepfakes are found only through a page or search result that connects them to a real photo of the user. Deepfakes posted on pages with no such connection are not returned by public reverse search (see the experiment above).
- **Page scan coverage.** Pages that load their images with JavaScript only, or require a login (most social networks), return no images, and at most 5 pages with 30 images each are scanned.
- **Reference photo.** Glasses, strong angles or heavy filters on the uploaded photo increase the face distance and can push deepfakes above the threshold.
- **Lookalikes.** Verification uses a single reference photo, so close lookalikes can pass the filter (15 of 18 lookalikes found through Yandex passed at distance about 0.39). Averaging over four reference photos with threshold 0.50 reduced this to 7 of 18 while keeping all deepfakes; this is not yet implemented.
- **Face swap residue.** A face swap keeps features of both people, so photos of the person whose image was the base of the swap can also pass the filter.
- **One manipulation type.** The retrained layer has seen only InsightFace style face swaps and real photos. Other swap methods, fully synthetic or diffusion generated faces, and heavily filtered photos are not reliably covered, and old, blurry or very small photos cause more false alarms.
- **Small personal test set.** Results on the author's data rest on 11 real photos and 4 deepfakes of one person, and one of the deepfakes is missed by the retrained detector; the 1,058 image test set complements this but covers other people.
- **Prototype scale.** One analysis at a time, with sequential deepfake checks and no job queue.
- **Not legal evidence.** The system does not decide whether a use is unlawful; the alteration indication is supporting, not conclusive.

## Citation

This software was developed for the following undergraduate dissertation:

> A. Kamilaki, "Development of a System for Detecting Unauthorized Use of Facial Images and Indicating Synthetic Alteration in Online Content," Undergraduate Dissertation, Department of Informatics, Democritus University of Thrace, Kavala, 2026.

## License

Copyright (c) 2026 Anna Kamilaki. All rights reserved. See [LICENSE](LICENSE) for permitted use.

## Author

Anna Kamilaki
