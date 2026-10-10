"""
Ryhmiseis tou systimatos efarmoghs.

Oles oi times diavazontai apo to arxeio .env wste na min yparxoun
kleidia (API keys) mesa ston kodika.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# --- Reverse image search (SerpApi) ---------------------------------------

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
SERPAPI_BASE = "https://serpapi.com/search"
SERPAPI_UPLOAD = "https://serpapi.com/image"

# Poies mhxanes anazhtishs tha xrisimopoihthoun.
SEARCH_ENGINES = ["google_lens"]

# Posa apotelesmata epistrefei kathe mhxanh.
MAX_RESULTS_PER_ENGINE = 40


# --- Face matching ---------------------------------------------------------

# Katwfli omoiotitas prosopou (cosine distance, DeepFace/Facenet512).
# Oso poio mikrh apostash toso poio omoiomorfo to prwsopo. Me 0.60 pernane
# ~93% twn diko mou fotografiwn kai ~91% twn deepfakes, enw ksenoi mono ~1.6%.
FACE_DISTANCE_THRESHOLD = 0.60

# Megisto megethos pixels eikonas prin to face encoding, gia taxytita.
MAX_IMAGE_DIMENSION = 1024

# Perithwrio gyrw apo to proswpo (0.3 = 30%) kai megethos tou crop
# pou stelnetai sto DeepSafe.
FACE_CROP_MARGIN = 0.3
FACE_CROP_SIZE = 256


# --- Deepfake detection (DeepSafe) -----------------------------------------

# Trexei topika me `make start` se diko tou fakelo.
DEEPSAFE_URL = os.getenv("DEEPSAFE_URL", "http://localhost:8000")

# voting, average, stacking
# Me ena montelo to average dinei thn pithanotita tou idiou tou montelou,
# enw to voting edine panta 0 h 1.
DEEPSAFE_ENSEMBLE = os.getenv("DEEPSAFE_ENSEMBLE", "average")

# Poia montela tou DeepSafe xrisimopoioume. To npr_deepfakedetection edine
# 0.0 se oles tis eikones tou test set, opote to vgalame apo thn psifoforia.
DEEPSAFE_MODELS = ["universalfakedetect"]


# --- Genika ----------------------------------------------------------------

# Timeout se deuterolepta gia ta exwterika API.
HTTP_TIMEOUT = 240.0

# Posa apotelesmata na katevainoun parallila.
DOWNLOAD_CONCURRENCY = 5
