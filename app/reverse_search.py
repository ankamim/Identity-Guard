"""
Reverse image search

Stelnei tin eikona tou xristi se mhxanes anazhtishs mesw SerpApi
kai epistrefei mia lista me upopsifia apotelesmata (URL eikonas + pigi).
"""

import asyncio
import io
import logging
from dataclasses import dataclass, field

import httpx
from PIL import Image

from . import config

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """Ena apotelesma apo tin antistrofi anazitisi."""

    image_url: str
    source_url: str = ""
    title: str = ""
    engine: str = ""
    # ta "" gemizoyn poio meta apo to face_matcher/deepfake_client
    face_distance: float | None = None
    is_face_match: bool = False
    deepfake_verdict: str | None = None
    deepfake_confidence: float | None = None
    model_results: dict = field(default_factory=dict)
    error: str | None = None


async def upload_image(client: httpx.AsyncClient, image_bytes: bytes,
                       filename: str) -> str:
    """
    Anevazei tin eikona sto SerpApi kai epistrefei ena image_id.

    To image_id xrhsimopoietai meta ws parametros sto Google Lens search,
    wste na mhn xreiazetai h eikona na einai idi dimosia sto internet.

    To SerpApi exei orio 500KB gia to arxeio, opote an h eikona einai megalyterh prepei prwnta na sympiestei.
    """
    image_bytes = _ensure_under_limit(image_bytes)

    files = {"image": (filename, image_bytes)}
    data = {"api_key": config.SERPAPI_KEY}

    response = await client.post(config.SERPAPI_UPLOAD, files=files, data=data)
    response.raise_for_status()
    payload = response.json()

    image_id = payload.get("image_id")
    if not image_id:
        raise RuntimeError(f"To SerpApi den epestrepse image_id: {payload}")
    return image_id


def _ensure_under_limit(image_bytes: bytes, limit: int = 480_000) -> bytes:
    """
    To SerpApi dexetai mexri 500KB. An h eikona einai megalyteri, kwdikopoieitai ksana
    ws JPEG me meiwmenh poiothta mexri na xoresei.
    """
    if len(image_bytes) <= limit:
        return image_bytes

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    for quality in (85, 70, 55, 40, 25):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        data = buffer.getvalue()
        if len(data) <= limit:
            logger.info("H eikona sympiesthke se %d bytes (quality=%d)",
                       len(data), quality)
            return data

    # An akoma den xwraei, mhkrainontai kai oi diastaseis
    image.thumbnail((800, 800))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=70)
    return buffer.getvalue()


def _parse_google_lens(payload: dict) -> list[SearchHit]:
    """Vgazei ta apotelesmata apo tin apantisi tou Google Lens."""
    hits: list[SearchHit] = []
    # "visual_matches" -> optika omoies eikones (default type)
    # "exact_matches" -> olohdies/scheddon-olohsies eikones (type=exact_matches)
    for key in ("visual_matches", "exact_matches"):
        for item in payload.get(key, []) or []:
            image_url = item.get("thumbnail") or item.get("image")
            if not image_url:
                continue
            hits.append(SearchHit(
                image_url=image_url,
                source_url=item.get("link", ""),
                title=item.get("title", ""),
                engine="google_lens",
            ))
    return hits


def _parse_yandex(payload: dict) -> list[SearchHit]:
    """Vgazei ta apotelesmata apo tin apanthsi tou Yandex."""
    hits: list[SearchHit] = []
    # To SerpApi epistrefei diaforetika keys analoga me to endpoint,
    # elegxoume ola ta pithana.
    for key in ("image_results", "similar_images", "organic_results"):
        for item in payload.get(key, []) or []:
            image_url = (item.get("original")
                         or item.get("thumbnail")
                         or item.get("image"))
            if not image_url:
                continue
            hits.append(SearchHit(
                image_url=image_url,
                source_url=item.get("link", "") or item.get("source", ""),
                title=item.get("title", ""),
                engine="yandex",
            ))
    return hits


async def _search_one_engine(client: httpx.AsyncClient, engine: str,
                             image_id: str) -> list[SearchHit]:
    """Kanei to aitima se mia sygkekrimeni mhxanh anazitisis."""
    params = {
        "engine": engine,
        "api_key": config.SERPAPI_KEY,
        "image_id": image_id,
    }

    try:
        response = await client.get(config.SERPAPI_BASE, params=params)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning("Apotyxia anazhtishs sto %s: %s", engine, exc)
        return []

    if payload.get("error"):
        logger.warning("To %s epestrepse sfalma: %s", engine, payload["error"])
        return []

    if engine == "google_lens":
        hits = _parse_google_lens(payload)
    else:
        hits = _parse_yandex(payload)

    return hits[:config.MAX_RESULTS_PER_ENGINE]


async def search(image_bytes: bytes, filename: str = "query.jpg") -> list[SearchHit]:
    """
    Kyria synarthsi tou module.

    Anevazei thn eikona mia fora kai meta psaxnei parallila se oles
    ths rythmismenes mhxanes anazitisis.
    """
    if not config.SERPAPI_KEY:
        raise RuntimeError(
            "Den vrethike SERPAPI_KEY. Vale to sto arxeio .env."
        )

    async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as client:
        image_id = await upload_image(client, image_bytes, filename)
        logger.info("Anevike h eikona, image_id=%s", image_id)

        tasks = [
            _search_one_engine(client, engine, image_id)
            for engine in config.SEARCH_ENGINES
        ]
        results = await asyncio.gather(*tasks)

    # Enwsh apotelesmatwn kai afairesi diplotypon me vash to URL
    all_hits: list[SearchHit] = []
    seen: set[str] = set()
    for engine_hits in results:
        for hit in engine_hits:
            if hit.image_url in seen:
                continue
            seen.add(hit.image_url)
            all_hits.append(hit)

    logger.info("Vrethhkan %d monadika apotelesmata", len(all_hits))
    return all_hits


async def download_images(hits: list[SearchHit]) -> dict[str, bytes]:
    """
    Katevazei ta arxeia eikonas gia na mporoun na analythoun.

    Epistrefei ena dictionary: image_url -> bytes.
    Osa apotyxoun apla paraleipontai.
    """
    semaphore = asyncio.Semaphore(config.DOWNLOAD_CONCURRENCY)
    downloaded: dict[str, bytes] = {}

    async def fetch(client: httpx.AsyncClient, hit: SearchHit) -> None:
        async with semaphore:
            try:
                response = await client.get(hit.image_url,
                                            follow_redirects=True)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if not content_type.startswith("image/"):
                    hit.error = "Den einai eikona"
                    return
                downloaded[hit.image_url] = response.content
            except Exception as exc:
                hit.error = f"Apotyxia download: {exc}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        await asyncio.gather(*(fetch(client, hit) for hit in hits))

    logger.info("Katevhkan %d apo %d eikones", len(downloaded), len(hits))
    return downloaded