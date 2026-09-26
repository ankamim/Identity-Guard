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
MAX_RESULTS_PER_ENGINE = 20


# --- Face matching ---------------------------------------------------------

# Katwfli omoiotitas prosopou (cosine distance, DeepFace/VGG-Face).
# Oso poio mikrh apostash toso poio omoiomorfo to prwsopo,typiki timi gia VGG-Face: ~0.40.
FACE_DISTANCE_THRESHOLD = 0.40

# Megisto megethos pixels eikonas prin to face encoding, gia taxytita.
MAX_IMAGE_DIMENSION = 1024


# --- Deepfake detection (DeepSafe) -----------------------------------------

# Trexei topika me `make start` se diko tou fakelo.
DEEPSAFE_URL = os.getenv("DEEPSAFE_URL", "http://localhost:8000")

# voting, average, stacking
DEEPSAFE_ENSEMBLE = os.getenv("DEEPSAFE_ENSEMBLE", "voting")


# --- Genika ----------------------------------------------------------------

# Timeout se deuterolepta gia ta exwterika API.
HTTP_TIMEOUT = 60.0

# Posa apotelesmata na katevainoun parallila.
DOWNLOAD_CONCURRENCY = 5
