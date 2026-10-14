"""
Deepfake detection mesw DeepSafe 

Kalei to DeepSafe (https://github.com/siddharthksah/DeepSafe) pou trexei
topika se Docker. To DeepSafe syndiazei polla montela opws NPR kai 
UniversalFakeDetect kai epistrefei thn enoieai apofash tous.
"""

import asyncio
import base64
import logging

import httpx

from . import config

logger = logging.getLogger(__name__)


async def check_health() -> bool:
    """Elegxei an to DeepSafe trexei prhn arxisei h analysi."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{config.DEEPSAFE_URL}/docs")
            return response.status_code < 500
    except Exception:
        return False


async def analyse_one(client: httpx.AsyncClient,
                      image_bytes: bytes) -> dict:
    """
    Stelnei mia eikona sto DeepSafe kai epistrefei to apotelesma.

    Xrhsimopoioume to /predict endpoint me base64, giati einai pio
    axiopisto apo to file upload otan h eikona einai stin mnhmh.
    """
    encoded = base64.b64encode(image_bytes).decode("ascii")

    payload = {
        "media_type": "image",
        "image_data": encoded,
        "ensemble_method": config.DEEPSAFE_ENSEMBLE,
        "models": config.DEEPSAFE_MODELS,
        "threshold": config.DEEPSAFE_THRESHOLD,
    }

    response = await client.post(
        f"{config.DEEPSAFE_URL}/predict",
        json=payload,
    )
    response.raise_for_status()
    return response.json()


async def analyse_batch(images: dict[str, bytes]) -> dict[str, dict]:
    """
    Analush pollwn eikonwn.

    Trexei seiriaka epeidi to DeepSafe trexei montela deep learning topika.
    parallhla aitimata to uperfortwnoun kanontas to poio argo

    Epistrefei: image_url -> apotelesma (h sfalma).
    """
    results: dict[str, dict] = {}

    async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as client:
        for url, image_bytes in images.items():
            try:
                results[url] = await analyse_one(client, image_bytes)
            except Exception as exc:
                logger.warning("Apotyxia analyshs gia %s: %s", url, exc)
                results[url] = {"error": str(exc)}
            # Mikri pafsh gia na min pnigei h ypiresia
            await asyncio.sleep(0.1)

    return results
