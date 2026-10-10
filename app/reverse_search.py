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
    """Ena apotelesma apo tin antistrofi anazhtisi."""
 
    image_url: str
    source_url: str = ""
    title: str = ""
    engine: str = ""
    #Plhroforia gia thn eikona meta to download (width, height, bytes)
    image_kind: str = "original"   # "original" | "thumbnail"
    img_width: int | None = None
    img_height: int | None = None
    img_bytes: int | None = None
    # Gemizei argotera apo to face_matcher / deepfake_client
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
 
    To image_id xrisimopoieitai meta ws parametros sto Google Lens search,
    wste na min xreiazetai i eikona na einai idi dimosia sto internet.
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
    To SerpApi dexetai mexri 500KB. An i eikona einai megalyteri, tin
    ksanakodikopoioume ws JPEG me meiomeni poiotita mexri na xoresei.
    """
    if len(image_bytes) <= limit:
        return image_bytes
 
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
 
    for quality in (85, 70, 55, 40, 25):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        data = buffer.getvalue()
        if len(data) <= limit:
            logger.info("I eikona sympiestike se %d bytes (quality=%d)",
                       len(data), quality)
            return data
 
    # An akoma den xorese, mikrainoume kai tis diastaseis
    image.thumbnail((800, 800))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=70)
    return buffer.getvalue()
 

def _as_url(value) -> str | None:
    """To SerpApi merikes fores epistrefei antikeimeno anti gia string."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("link", "url", "src", "original", "thumbnail"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
        logger.warning("Agnwsto sxhma URL eikonas: %s", str(value)[:200])
    return None

def _parse_google_lens(payload: dict) -> list[SearchHit]:
    """Vgazei ta apotelesmata apo tin apantisi tou Google Lens."""
    hits: list[SearchHit] = []
    # "visual_matches" -> optika omoies eikones (default type)
    # "exact_matches" -> tautosimes/scheddon-tautosimes eikones (type=exact_matches)
    for key in ("visual_matches", "exact_matches"):
        for item in payload.get(key, []) or []:
            image_url = (_as_url(item.get("original") or item.get("image") or item.get("thumbnail"))
                         or _as_url(item.get("image"))
                         or _as_url(item.get("thumbnail")))
            if not image_url:
                continue
            # Ta exact_matches dinoun mono mikrografia (~225px), opou h
            # anixneush einai ligotero axiopisti. To grafoume sthn anafora.
            has_original = _as_url(item.get("original")) or _as_url(item.get("image"))
            hits.append(SearchHit(
                image_url=image_url,
                source_url=item.get("link", ""),
                title=item.get("title", ""),
                engine="google_lens",
                image_kind="original" if has_original else "thumbnail",
            ))
    return hits
 
 
def _parse_yandex(payload: dict) -> list[SearchHit]:
    """Vgazei ta apotelesmata apo tin apantisi tou Yandex."""
    hits: list[SearchHit] = []
    # To SerpApi epistrefei diaforetika keys analoga me to endpoint,
    # opote elegxoume ola ta pithana.
    for key in ("image_results", "similar_images", "organic_results"):
        for item in payload.get(key, []) or []:
            image_url = (_as_url(item.get("original"))
                         or _as_url(item.get("image"))
                         or _as_url(item.get("thumbnail")))
            if not image_url:
                continue
            has_original = _as_url(item.get("original")) or _as_url(item.get("image"))
            hits.append(SearchHit(
                image_url=image_url,
                source_url=item.get("link", "") or item.get("source", ""),
                title=item.get("title", ""),
                engine="yandex",
                image_kind="original" if has_original else "thumbnail",
            ))
    return hits
 
 
async def _search_one_engine(client: httpx.AsyncClient, engine: str,
                             image_id: str) -> list[SearchHit]:
    """Kanei to aitima se mia sygkekrimeni mihani anazitisis."""
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
        logger.warning("Apotyxia anazitisis sto %s: %s", engine, exc)
        return []
 
    if payload.get("error"):
        logger.warning("To %s epestrepse sfalma: %s", engine, payload["error"])
        return []
 
    if engine == "google_lens":
        hits = _parse_google_lens(payload)
    else:
        hits = _parse_yandex(payload)

    logger.info("Apotelesmata prin to orio: %d", len(hits))
    return hits[:config.MAX_RESULTS_PER_ENGINE]
 
 
async def _search_exact_matches(client: httpx.AsyncClient,
                                image_id: str) -> list[SearchHit]:
    """
    Xexoristo aitima sto Google Lens me type=exact_matches.
 
    To default search (visual_matches) vriskei optika omoies eikones,
    pou den einai to idio me "i idia fotografia emfanizetai allou".
    To exact_matches einai pio kontino se afto pou psaxnoume gia tin
    ptyxiaki: pou allou emfanizetai AFTI i sygkekrimeni eikona.
    """
    params = {
        "engine": "google_lens",
        "api_key": config.SERPAPI_KEY,
        "image_id": image_id,
        "type": "exact_matches",
    }
 
    try:
        response = await client.get(config.SERPAPI_BASE, params=params)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning("Apotyxia exact_matches anazitisis: %s", exc)
        return []
 
    if payload.get("error"):
        logger.warning("To exact_matches epestrepse sfalma: %s", payload["error"])
        return []
 
    hits = _parse_google_lens(payload)
    
    logger.info("Apotelesmata exact_matches: %d", len(hits))
    return hits[:config.MAX_RESULTS_PER_ENGINE]

async def _search_yandex_by_url(client: httpx.AsyncClient,
                                public_image_url: str) -> list[SearchHit]:
    """
    Anazhtish Yandex mesw dimosiou URL tis eikonas.
 
    To Yandex sto SerpApi DEN dexetai image_id (upload) opos to Google Lens -
    thelei mono "url" parametro me idi dimosia prosvasimi eikona.
    """
    params = {
        "engine": "yandex_images",
        "api_key": config.SERPAPI_KEY,
        "url": public_image_url,
    }
 
    try:
        response = await client.get(config.SERPAPI_BASE, params=params)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning("Apotyxia anazitisis sto yandex (url): %s", exc)
        return []
 
    if payload.get("error"):
        logger.warning("To yandex epestrepse sfalma: %s", payload["error"])
        return []
 
    hits = _parse_yandex(payload)
    
    logger.info("Apotelesmata yandex: %d", len(hits))
    return hits[:config.MAX_RESULTS_PER_ENGINE]
 
 
async def search(image_bytes: bytes, filename: str = "query.jpg", public_image_url: str | None = None) -> list[SearchHit]:
    """
    Kyria synartisi tou module.
 
    Anevazei tin eikona mia fora kai meta psaxnei parallila se oles
    tis rythmismenes mihanes anazitisis.
    """
    if not config.SERPAPI_KEY:
        raise RuntimeError(
            "Den vrethike SERPAPI_KEY. Valto sto arxeio .env."
        )
 
    async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as client:
        image_id = await upload_image(client, image_bytes, filename)
        logger.info("Anevike i eikona, image_id=%s", image_id)
 
        tasks = [
            _search_one_engine(client, engine, image_id)
            for engine in config.SEARCH_ENGINES
        ]
        if "google_lens" in config.SEARCH_ENGINES:
            tasks.append(_search_exact_matches(client, image_id))
        if public_image_url:
            tasks.append(_search_yandex_by_url(client, public_image_url))
        results = await asyncio.gather(*tasks)
        logger.info("Apotelesmata ana anazhthsh: %s", [len(r) for r in results])
 
    # Enopoihsh kai afairesh diplotypon me vasi to URL
    all_hits: list[SearchHit] = []
    seen: set[str] = set()
    for engine_hits in results:
        for hit in engine_hits:
            if hit.image_url in seen:
                continue
            seen.add(hit.image_url)
            all_hits.append(hit)
 
    logger.info("Vrethikan %d monadika apotelesmata", len(all_hits))
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
                hit.error = f"Apotyxia katevasmatos: {exc}"
 
    async with httpx.AsyncClient(timeout=30.0) as client:
        await asyncio.gather(*(fetch(client, hit) for hit in hits))
 
    logger.info("Katevikan %d apo %d eikones", len(downloaded), len(hits))
    return downloaded