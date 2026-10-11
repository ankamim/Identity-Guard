# Identity Guard

This repository contains the software for a thesis on detecting the unauthorized use of a person's facial images online. Identity Guard is a web application: the user uploads one photo of their face, the system searches publicly indexed sources for images that contain the same person, confirms each candidate with face verification, and passes the confirmed matches to a deepfake detector that indicates whether the face may have been synthetically altered (face swap). The result is a report listing where the face appears, how close each match is, and which matches look manipulated.

## Motivation

Photos posted on social media can be reused without consent on fake profiles, scam pages, or in manipulated images. Reverse image search engines find visually similar pictures, but they return many images of other people and give no indication of manipulation. This project combines three existing building blocks into one pipeline: reverse image search (Google Lens through SerpApi), face verification (DeepFace with Facenet512) to keep only images of the same person, and a deepfake detector (UniversalFakeDetect served by DeepSafe) to flag likely face swaps.

The off-the-shelf detector was not trained on face swaps and caught only part of them. Its final linear layer was therefore retrained on face swaps from DeepFakeFace and on real faces from DeepFakeFace and FFHQ. On 1,058 unseen test images this raised accuracy from 81.3% to 92.9%, with 95.3% of face swaps caught and 8.5% of real photos flagged. The scripts that produced these weights are included in `retraining/`.

## Repository Structure

```
Identity-Guard/
├── app/
│   ├── config.py               # Thresholds, search engines, DeepSafe settings (keys read from .env)
│   ├── reverse_search.py       # Module 1: reverse image search through SerpApi, image download
│   ├── face_matcher.py         # Module 2: face detection, Facenet512 embeddings, distance, face crop
│   ├── deepfake_client.py      # Module 3: calls to the DeepSafe /predict API
│   ├── pipeline.py             # Module 4: runs the stages in order and builds the report
│   ├── main.py                 # FastAPI server (web page, /api/status, /api/analyse)
│   └── static/index.html       # User interface
├── models/
│   └── fc_weights_faceswap.pth # Retrained final layer for UniversalFakeDetect
├── retraining/                 # Scripts and training log that produced the weights above
│   ├── make_crops.py
│   ├── make_crops2.py
│   ├── extract_feats.py
│   ├── train_fc.py
│   ├── install_fc.py
│   ├── verify_api.py
│   └── train_fc_log.txt
├── compare_engines.py          # Google Lens vs Yandex comparison for one public image
├── measure_search.py           # Search and face matching measurements (no DeepSafe)
├── deepsafe_check.py           # Standalone DeepSafe test, optionally on degraded copies
├── sanity_check.py             # Sends one image straight to DeepSafe
├── start-identity-guard.bat    # Windows launcher for DeepSafe and the app
├── requirements.txt
└── .env.example
```

## How It Works

1. **Reference face.** The uploaded photo is checked for exactly one face, and a Facenet512 embedding is computed for it (DeepFace, MTCNN detector). If no face or several faces are found, the request stops here, before any search quota is used.
2. **Reverse image search.** The photo is uploaded to SerpApi and searched on Google Lens, both for visually similar images and for exact matches. When a public URL of the photo is available (as in `compare_engines.py`), Yandex is searched as well.
3. **Download.** The candidate images are downloaded in parallel and kept in memory only.
4. **Face filter.** Each candidate's face is compared with the reference by cosine distance. Only candidates under `FACE_DISTANCE_THRESHOLD` (0.60) count as the same person.
5. **Deepfake check.** For every confirmed match, the face is cropped (30% margin, 256 px) and sent to DeepSafe, where UniversalFakeDetect with the retrained layer returns a probability of manipulation. Values at or above `DEEPSAFE_THRESHOLD` (0.24) are flagged as likely synthetic. Compressed Yandex thumbnails are skipped because they caused false alarms on real photos.
6. **Report.** The web page shows each confirmed match with its source, similarity, and deepfake verdict, together with counts for every stage and the total time.

## Module Reference

### `app/config.py`
All settings in one place: search engines and results per engine, the face distance threshold, image size limits, face crop margin and size, DeepSafe URL, ensemble method, models and threshold, HTTP timeout, and download concurrency. The SerpApi key and DeepSafe URL are read from `.env`.

### `app/reverse_search.py`
Uploads the query image to SerpApi (shrinking it below the upload limit when needed), runs Google Lens searches for similar images and exact matches, optionally searches Yandex by public URL, parses each engine's response into `SearchHit` records, and downloads the candidate images concurrently.

### `app/face_matcher.py`
Wraps DeepFace. Detects faces with MTCNN, rejects query photos with no face or several faces, computes Facenet512 embeddings, compares a candidate with the reference by cosine distance (using the closest face when a candidate has several), and crops the matching face for the deepfake check.

### `app/deepfake_client.py`
Client for the DeepSafe API. Checks that the service is up, sends each face crop to `/predict` with the configured models and ensemble method, applies `DEEPSAFE_THRESHOLD` to the returned probability, and runs a batch of images concurrently.

### `app/pipeline.py`
Runs the stages above in order for one uploaded photo, records counts at each stage (found, downloaded, containing a face, face matches, flagged as synthetic), and builds the JSON report returned to the interface.

### `app/main.py`
FastAPI application. Serves the interface, reports through `/api/status` whether SerpApi and DeepSafe are configured and reachable, and accepts uploads on `/api/analyse` (JPEG, PNG or WebP, up to 10 MB).

### `retraining/`
Retrains only the final linear layer (768 weights and a bias) of UniversalFakeDetect on top of its frozen CLIP ViT-L/14 features.

- `make_crops.py` crops faces exactly as the app does, from DeepFakeFace split by identity folder (00 to 69 train, 70 to 84 validation, 85 to 99 test), plus the author's own test photos.
- `make_crops2.py` adds FFHQ real faces and a degraded copy (rescaling, JPEG compression, light blur) of half of the training crops.
- `extract_feats.py` runs inside the DeepSafe container and saves the CLIP features of every crop.
- `train_fc.py` trains an L2 regularized logistic regression, chooses the regularization and threshold on the validation split only, and evaluates the test split once.
- `install_fc.py` converts the result into the model's weight file inside the container.
- `verify_api.py` checks the installed weights through the real DeepSafe API.
- `train_fc_log.txt` is the output of the run that produced `models/fc_weights_faceswap.pth`.

### Experiment scripts
`compare_engines.py`, `measure_search.py`, `deepsafe_check.py` and `sanity_check.py` were used for the measurements in the thesis. Each exercises one part of the pipeline (search engines, face matching, DeepSafe) without the web interface.

## Running the Application

Python 3.10 or newer, Docker, and a SerpApi key (the free plan allows 250 searches per month) are required.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your SERPAPI_KEY
```

Start DeepSafe in a separate folder. The current DeepSafe image does not install OpenCV for UniversalFakeDetect, so first add `opencv-python-headless==4.11.0.86` to `models/image/universalfakedetect/requirements.txt`:

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

Open http://localhost:8080 (DeepSafe listens on port 8000). The weights must be copied again whenever the container is rebuilt, and the 0.24 threshold is only valid with these weights. Each analysis uses two SerpApi searches.

## Reproducing the Retrained Detector

The datasets are not included because of their size and licenses:

- [DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace): `wiki.zip` (real) and `insight.zip` (face swaps), extracted to `dff/wiki` and `dff/insight`.
- [FFHQ 512](https://huggingface.co/datasets/Ryan-sjtu/ffhq512-caption): the first parquet file, used as additional real faces.

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

Feature extraction takes about 2 hours on a CPU for roughly 5,300 crops. Results on the test split, from `retraining/train_fc_log.txt`:

| Test set (1,058 images) | Original layer @ 0.06 | Retrained layer @ 0.24 |
|---|---|---|
| AUC | 0.870 | 0.987 |
| Accuracy | 81.3% | 92.9% |
| Face swaps caught | 75.3% | 95.3% |
| Real photos flagged | 15.3% | 8.5% |
| Author's own photos (15) | 12/15 | 15/15 |

## Limitations

- Google Lens has no official API, so searches go through a third party provider (SerpApi), which adds a dependency and a cost per search.
- Only publicly indexed content is covered. Private social media profiles are not reachable.
- The deepfake indication is probabilistic. The retrained layer has seen only InsightFace style face swaps and real photos; other manipulation methods, fully synthetic faces and heavily filtered photos are not reliably covered, and old, blurry or very small photos cause more false alarms.
- Face verification uses a single reference photo, so close lookalikes can pass the face filter.
- Third party images are processed in memory and not stored.

## Citation

This software was developed for the following undergraduate dissertation:

> A. Kamilaki, "Development of a System for Detecting Unauthorized Use of Facial Images and Indicating Synthetic Alteration in Online Content," Undergraduate Dissertation, Department of Informatics, Democritus University of Thrace, Kavala, 2026.

## License

Copyright (c) 2026 Anna Kamilaki. All rights reserved. See [LICENSE](LICENSE) for permitted use.

## Author

Anna Kamilaki
