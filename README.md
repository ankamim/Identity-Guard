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

**Αν το `face_recognition` δεν εγκαθίσταται** (συνηθισμένο σε Windows —
θέλει το dlib, που απαιτεί C++ compiler), έχεις δύο επιλογές:

- Εγκατάσταση των Visual Studio Build Tools και CMake, μετά ξανά `pip install`
- Αντικατάσταση με `deepface` (καθαρό pip install, χωρίς compiler). Θα
  χρειαστεί να ξαναγράψεις το `face_matcher.py`, αλλά η λογική είναι ίδια:
  embedding → απόσταση → κατώφλι.

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
| `SEARCH_ENGINES` | Ποιες μηχανές χρησιμοποιούνται |
| `MAX_RESULTS_PER_ENGINE` | Πόσα αποτελέσματα ανά μηχανή — προσοχή, κάθε αναζήτηση καίει quota |
| `DEEPSAFE_ENSEMBLE` | `voting`, `average` (προεπιλογή) ή `stacking` |
| `DEEPSAFE_MODELS` | Ποια μοντέλα του DeepSafe καλούνται. Μόνο το `universalfakedetect`, το `npr_deepfakedetection` έδινε 0.0 σε όλες τις εικόνες. |
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
- Η ένδειξη αλλοίωσης είναι πιθανοτική. Τα μοντέλα του DeepSafe έχουν
  εκπαιδευτεί σε συγκεκριμένους generators και γενικεύουν ατελώς.
- Το σύστημα κατεβάζει και επεξεργάζεται εικόνες τρίτων προσωρινά στη
  μνήμη. Δεν αποθηκεύεται τίποτα, αλλά αξίζει συζήτηση GDPR στο κεφάλαιο
  ηθικής.

## Επόμενα βήματα

1. Εξαγωγή της αναφοράς σε PDF
2. Ελεγχόμενο test set με ground truth για μέτρηση precision/recall
3. Caching αποτελεσμάτων ώστε οι επαναλαμβανόμενες δοκιμές να μην καίνε quota
