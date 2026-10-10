# Identity Guard

Πρωτότυπο σύστημα εντοπισμού μη εξουσιοδοτημένης χρήσης εικόνων προσώπου,
με βοηθητική ένδειξη ψηφιακής αλλοίωσης.

Ο χρήστης ανεβάζει μία φωτογραφία. Το σύστημα ψάχνει σε δημόσια
ευρετηριασμένες πηγές, επιβεβαιώνει ποια ευρήματα περιέχουν όντως το ίδιο
πρόσωπο, και περνάει τα επιβεβαιωμένα από έναν έτοιμο ανιχνευτή deepfake.

## Δομή

```
app/
  config.py           ρυθμίσεις και κατώφλια
  reverse_search.py   Module 1 — αναζήτηση μέσω SerpApi
  face_matcher.py     Module 2 — επαλήθευση προσώπου
  deepfake_client.py  Module 3 — κλήση στο DeepSafe
  pipeline.py         Module 4 — ενορχήστρωση
  main.py             FastAPI server
  static/index.html   διεπαφή
```

## Εγκατάσταση

### 1. Εξαρτήσεις

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Η επαλήθευση προσώπου γίνεται με το `deepface` (Facenet512, ανιχνευτής
MTCNN), καθαρό pip install χωρίς compiler. Θέλει Python 3.10 ή νεότερη.

### 2. Κλειδί SerpApi

Κάνε εγγραφή στο serpapi.com (δωρεάν πλάνο: 250 αναζητήσεις/μήνα), πάρε το
κλειδί, και:

```bash
cp .env.example .env
# άνοιξε το .env και βάλε το κλειδί
```

### 3. DeepSafe

Σε ξεχωριστό φάκελο:

```bash
git clone https://github.com/siddharthksah/DeepSafe.git
cd DeepSafe
make start
```

Πιάνει το port 8000. Έλεγξε ότι δουλεύει: http://localhost:8000/docs

**Προσοχή:** η τρέχουσα έκδοση του DeepSafe δεν εγκαθιστά το OpenCV στο
`universalfakedetect` και το container κάνει συνέχεια restart
(`No module named 'cv2'`). Πριν το `make start`, πρόσθεσε τη γραμμή
`opencv-python-headless==4.11.0.86` στο
`models/image/universalfakedetect/requirements.txt`. Η εφαρμογή χρησιμοποιεί
μόνο το `universalfakedetect`, οπότε αρκεί και
`docker compose up -d --build api universalfakedetect` (χωρίς `make`).

Το αρχικό `universalfakedetect` δεν έχει εκπαιδευτεί σε face swaps. Το
τελευταίο του επίπεδο ξαναεκπαιδεύτηκε σε face swaps του DeepFakeFace και σε
πραγματικές φωτογραφίες (DeepFakeFace, FFHQ). Μετά την εκκίνηση, από τον
φάκελο αυτού του repo:

```bash
docker cp models/fc_weights_faceswap.pth deepsafe-universalfakedetect:/app/model/universalfakedetect/pretrained_weights/fc_weights.pth
docker restart deepsafe-universalfakedetect
```

Χρειάζεται ξανά μετά από κάθε `--build` ή αναδημιουργία του container. Το
`DEEPSAFE_THRESHOLD` (0.24) ισχύει μόνο με αυτά τα βάρη.

### 4. Εκκίνηση

```bash
uvicorn app.main:app --reload --port 8080
```

Άνοιξε http://localhost:8080

## Παράμετροι για πειραματισμό

Στο `config.py`:

| Παράμετρος | Τι κάνει |
|---|---|
| `FACE_DISTANCE_THRESHOLD` | Πόσο κοντά πρέπει να είναι δύο πρόσωπα για match. Μικρότερο = αυστηρότερο, λιγότερα false positives αλλά και λιγότερα σωστά ευρήματα. |
| `SEARCH_ENGINES` | Ποιες μηχανές χρησιμοποιούνται. Το `google_lens` κάνει δύο αναζητήσεις SerpApi (παρόμοιες και ίδιες εικόνες). Αν η εικόνα έχει δημόσιο URL (π.χ. μέσω ngrok), γίνεται αυτόματα και αναζήτηση στο Yandex, άρα τρεις ανά έλεγχο. |
| `MAX_RESULTS_PER_ENGINE` | Πόσα αποτελέσματα ανά μηχανή — προσοχή, κάθε αναζήτηση καίει quota |
| `DEEPSAFE_ENSEMBLE` | `voting`, `average` (προεπιλογή) ή `stacking` |
| `DEEPSAFE_MODELS` | Ποια μοντέλα του DeepSafe καλούνται. Μόνο το `universalfakedetect`, το `npr_deepfakedetection` έδινε 0.0 σε όλες τις εικόνες. |
| `DEEPSAFE_THRESHOLD` | Από ποια πιθανότητα και πάνω μια εικόνα θεωρείται αλλοιωμένη. Μικρότερο = πιάνει περισσότερα deepfakes αλλά δίνει και περισσότερους λάθος συναγερμούς. |
| `FACE_CROP_MARGIN`, `FACE_CROP_SIZE` | Περιθώριο και μέγεθος του κομμένου προσώπου που στέλνεται στο DeepSafe |

Το `FACE_DISTANCE_THRESHOLD` είναι το πιο ενδιαφέρον για την πτυχιακή: τρέξε
την αξιολόγηση με διάφορες τιμές (0.50, 0.55, 0.60, 0.65) και φτιάξε καμπύλη
precision/recall. Αυτό είναι ένα ολόκληρο υποκεφάλαιο αποτελεσμάτων.

## Περιορισμοί προς αναφορά στην εργασία

- Το Google Lens δεν έχει επίσημο API· η πρόσβαση γίνεται μέσω third-party
  παρόχου (SerpApi). Αυτό εισάγει εξάρτηση και κόστος.
- Καλύπτεται μόνο δημόσια ευρετηριασμένο περιεχόμενο. Ιδιωτικά προφίλ
  Instagram/Facebook δεν είναι προσβάσιμα, και το scraping τους θα
  παραβίαζε τους όρους χρήσης.
- Η ένδειξη αλλοίωσης είναι πιθανοτική. Το τελευταίο επίπεδο του
  `universalfakedetect` εκπαιδεύτηκε μόνο σε face swaps τύπου InsightFace
  (DeepFakeFace) και σε πραγματικές φωτογραφίες (DeepFakeFace, FFHQ). Άλλες
  μέθοδοι swap, πλήρως συνθετικά πρόσωπα και φωτογραφίες με έντονα φίλτρα
  δεν καλύπτονται με βεβαιότητα. Παλιές, θολές ή πολύ μικρές φωτογραφίες
  δίνουν περισσότερους λάθος συναγερμούς.
- Το σύστημα κατεβάζει και επεξεργάζεται εικόνες τρίτων προσωρινά στη
  μνήμη. Δεν αποθηκεύεται τίποτα, αλλά αξίζει συζήτηση GDPR στο κεφάλαιο
  ηθικής.

## Επόμενα βήματα

1. Εξαγωγή της αναφοράς σε PDF
2. Ελεγχόμενο test set με ground truth για μέτρηση precision/recall
3. Caching αποτελεσμάτων ώστε οι επαναλαμβανόμενες δοκιμές να μην καίνε quota
