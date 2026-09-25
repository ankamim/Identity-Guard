"""
To web server tis efarmoghs.

Trexei me:
    uvicorn app.main:app --reload --port 8080

http://localhost:8080

To DeepSafe pianei to 8000, gi' afto ebala 8080.
"""

import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, deepfake_client, face_matcher, pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Identity Guard")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def status() -> dict:
    """
    Elegxei an ola ta kommatia einai etoima.
    To frontend to kalei sthn arxh gia pithanh proeidopoihsh.
    """
    return {
        "serpapi_configured": bool(config.SERPAPI_KEY),
        "deepsafe_available": await deepfake_client.check_health(),
        "face_threshold": config.FACE_DISTANCE_THRESHOLD,
        "engines": config.SEARCH_ENGINES,
    }


@app.post("/api/analyse")
async def analyse(file: UploadFile = File(...)) -> dict:
    """Dexetai mia fotografia kai epistrefei thn anafora."""

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Dekta formats: JPEG, PNG, WebP.",
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail="H eikona einai megalyteri apo 10 MB.",
        )

    try:
        return await pipeline.run(image_bytes, file.filename or "query.jpg")

    except face_matcher.NoFaceFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except face_matcher.MultipleFacesError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        # px leipei to SERPAPI_KEY
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Aprosmeno sfalma sthn analysh")
        raise HTTPException(
            status_code=500,
            detail="H analysh apetyxe. Des ta logs tou server.",
        ) from exc
