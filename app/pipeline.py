"""
Pipeline

Syndeei ta 3 prohgoumena modules se mia rwh:

    eikona -> reverse search -> filtro prosopou -> deepfake -> anafora

Kratame stathstika se kathe stadio gia arxeio apotelesmatwn.
"""

import io
import logging
import time
from dataclasses import asdict
from PIL import Image
 
from . import deepfake_client, face_matcher, reverse_search
 
logger = logging.getLogger(__name__)
 
 
async def run(image_bytes: bytes, filename: str,
              public_image_url: str | None = None) -> dict:
    """
    Trexei oli tin analysi gia mia eikona.
 
    public_image_url: an dothei, xrisimopoieitai gia na
    psaxtei kai sto Yandex, pou apaitei dimosio URL anti gia upload.
    
    Epistrefei ena dictionary etoimo gia serialization se JSON.
    """
    started = time.time()
    stats = {
        "found_by_search": 0,
        "downloaded": 0,
        "contain_a_face": 0,
        "face_matches": 0,
        "flagged_synthetic": 0,
    }

    # --- Stadio 1: encoding ths eikonas anaforas -------------------------
    # Afto ginetai prwta. An apotyxei, den exei noima na psaxoume.
    query_encoding = face_matcher.encode_query_face(image_bytes)
    logger.info("Etoimo to encoding ths eikonas anaforas")

    # --- Stadio 2: reverse image search ----------------------------------
    hits = await reverse_search.search(image_bytes, filename, public_image_url)
    stats["found_by_search"] = len(hits)

    if not hits:
        return _build_report([], stats, started)

    # --- Stadio 3: download twn ypopsifion eikonon ----------------------
    downloaded = await reverse_search.download_images(hits)
    stats["downloaded"] = len(downloaded)

    # --- Stadio 4: filtro proswpou ---------------------------------------
    # Edw phgainoun osa den periexoun to idio proswpo.
    confirmed: list[reverse_search.SearchHit] = []
 
    for hit in hits:
        image_data = downloaded.get(hit.image_url)
    
        if image_data is None:
            continue
        try:
            with Image.open(io.BytesIO(image_data)) as im:
                hit.img_width, hit.img_height = im.size
        except Exception:
            pass
        hit.img_bytes = len(image_data)
 
        is_match, distance = face_matcher.compare_with_candidate(
            query_encoding, image_data
        )
 
        if distance is not None:
            stats["contain_a_face"] += 1
            hit.face_distance = round(distance, 4)
            logger.info("Apostasi gia %s: %.4f (match=%s)",
                       hit.image_url[:60], distance, is_match)
        else:
            logger.info("Den vrethike prosopo sto: %s", hit.image_url[:60])
 
        hit.is_face_match = is_match
        if is_match:
            confirmed.append(hit)
 
    stats["face_matches"] = len(confirmed)
    logger.info("Epivevaiothikan %d antistoixies prosopou", len(confirmed))

    # --- Stadio 5: deepfake detection ------------------------------------
    # Trexei mono panw sta epivevaiomena matches, oxi se ola.
    if confirmed:
        to_analyse = {
            hit.image_url: downloaded[hit.image_url]
            for hit in confirmed
        }
        deepfake_results = await deepfake_client.analyse_batch(to_analyse)
 
        for hit in confirmed:
            result = deepfake_results.get(hit.image_url, {})
            if "error" in result:
                hit.error = result["error"]
                continue
            hit.deepfake_verdict = result.get("verdict")
            hit.deepfake_confidence = result.get("confidence_in_verdict")
            hit.model_results = result.get("model_results", {})
            logger.info("DeepSafe gia %s: verdict=%s conf=%s raw=%s",
                       hit.image_url[:60], hit.deepfake_verdict,
                       hit.deepfake_confidence, hit.model_results)
            if hit.deepfake_verdict == "fake":
                stats["flagged_synthetic"] += 1
 
    return _build_report(confirmed, stats, started)
 
 
def _build_report(hits: list, stats: dict, started: float) -> dict:
    """Suntaksh telikhs anaforas"""
    findings = []
    for hit in hits:
        entry = asdict(hit)
        if hit.face_distance is not None:
            entry["similarity_percent"] = face_matcher.distance_to_similarity(
                hit.face_distance
            )
        findings.append(entry)
 
    # Taksinomish: prwta ta pio omoia proswpa
    findings.sort(key=lambda item: item.get("face_distance") or 1.0)
 
    return {
        "statistics": stats,
        "findings": findings,
        "duration_seconds": round(time.time() - started, 1),
    }