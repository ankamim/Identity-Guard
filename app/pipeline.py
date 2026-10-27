"""
Pipeline

Syndeei ta prohgoumena modules se mia rwh:

    eikona -> reverse search -> filtro prosopou -> sarwsh selidwn
           -> deepfake -> anazhthsh me ta deepfakes -> anafora

Kratame stathstika se kathe stadio gia arxeio apotelesmatwn.
"""

import io
import logging
import time
from dataclasses import asdict

import numpy as np
from PIL import Image

from . import config, deepfake_client, face_matcher, page_scan, reverse_search

logger = logging.getLogger(__name__)


async def run(images: list[tuple[bytes, str]],
              public_image_url: str | None = None) -> dict:
    """
    Trexei oli tin analysi gia mia h perissoteres fwtografies tou xrhsth.

    images: lista apo (bytes, onoma arxeiou), mexri MAX_QUERY_IMAGES.
    Kathe fwtografia psaxnetai xexwrista, giati to Lens vriskei selides
    me thn idia h paromoia eikona kai oxi me to idio proswpo.

    public_image_url: an dothei, xrisimopoieitai gia na
    psaxtei kai sto Yandex, pou apaitei dimosio URL anti gia upload.

    Epistrefei ena dictionary etoimo gia serialization se JSON.
    """
    started = time.time()
    stats = {
        "query_images": len(images),
        "found_by_search": 0,
        "downloaded": 0,
        "contain_a_face": 0,
        "page_images": 0,
        "traced_deepfakes": 0,
        "trace_results": 0,
        "face_matches": 0,
        "flagged_synthetic": 0,
    }

    # --- Stadio 1: encoding twn eikonwn anaforas -------------------------
    # Afto ginetai prwta. An apotyxei, den exei noima na psaxoume.
    query_encoding = [face_matcher.encode_query_face(image_bytes)
                      for image_bytes, _ in images]
    if len(query_encoding) == 1:
        query_encoding = query_encoding[0]
    logger.info("Etoima ta encodings %d eikonwn anaforas", len(images))

    # --- Stadio 2: reverse image search ----------------------------------
    # Mia anazhthsh ana fwtografia, kai enwnoume ta apotelesmata xwris
    # diplotypa. To Yandex thelei dimosio URL, opote mono gia thn prwth.
    hits = []
    seen = set()
    for number, (image_bytes, filename) in enumerate(images):
        image_hits = await reverse_search.search(
            image_bytes, filename, public_image_url if number == 0 else None)
        for hit in image_hits:
            if hit.image_url in seen:
                continue
            seen.add(hit.image_url)
            hit.query_image = number + 1
            hits.append(hit)
    stats["found_by_search"] = len(hits)

    if not hits:
        return _build_report([], stats, started)

    # --- Stadio 3: download twn ypopsifion eikonon ----------------------
    downloaded = await reverse_search.download_images(hits)
    stats["downloaded"] = len(downloaded)

    # --- Stadio 4: filtro proswpou ---------------------------------------
    # Edw phgainoun osa den periexoun to idio proswpo.
    confirmed = _filter_faces(hits, downloaded, query_encoding, stats)
    logger.info("Epivevaiothikan %d antistoixies prosopou", len(confirmed))

    # --- Stadio 5: sarwsh selidwn ----------------------------------------
    # Ta deepfakes enos prosopou einai alles eikones, opote to reverse
    # search den ta epistrefei. Psaxnoume stis selides twn matches (eite
    # pragmatikh fwtografia eite deepfake). Den kostizei anazhthseis.
    known = {hit.image_url for hit in hits}
    if confirmed and config.PAGE_SCAN_ENABLED:
        page_hits = await page_scan.scan_pages(confirmed, known)
        page_downloaded = await reverse_search.download_images(page_hits)
        stats["page_images"] = len(page_downloaded)
        downloaded.update(page_downloaded)
        known.update(hit.image_url for hit in page_hits)
        confirmed += _filter_faces(page_hits, page_downloaded, query_encoding,
                                   stats, min_side=config.PAGE_SCAN_MIN_SIDE,
                                   closest=True)

    # --- Stadio 6: deepfake detection ------------------------------------
    # Trexei mono panw sta epivevaiomena matches, oxi se ola.
    await _check_deepfakes(confirmed, downloaded, query_encoding, stats)

    # --- Stadio 7: pou allou yparxoun ta deepfakes -----------------------
    # Gia kathe deepfake kanoume nea antistrofi anazhthsh, gia na vroume
    # poies alles selides to anartisan. Kostizei 2 anazhthseis SerpApi ana
    # deepfake, gi' afto yparxei orio.
    fakes = [hit for hit in confirmed if hit.deepfake_verdict == "fake"]
    for fake in fakes[:config.DEEPFAKE_TRACE_MAX]:
        traced = await _trace_deepfake(fake, downloaded, known,
                                       query_encoding, stats)
        await _check_deepfakes(traced, downloaded, query_encoding, stats)
        confirmed += traced

    stats["face_matches"] = len(confirmed)
    return _build_report(confirmed, stats, started)


async def _check_deepfakes(hits: list, downloaded: dict, query_encoding,
                           stats: dict) -> None:
    """
    Stelnei ta epivevaiomena matches sto DeepSafe. Stelnoume mono to
    kommeno proswpo, an den vrethei h olokliri eikona.
    """
    to_analyse = {}
    for hit in hits:
        # Oi mikrografies tou Yandex (similar_images) einai sympiesmenes
        # kai edwsan lathos "fake" se pragmatikes fwtografies, opote
        # den tis elegxoume. Ta exact_matches tou Lens ta kratame.
        if hit.engine == "yandex" and hit.image_kind == "thumbnail":
            continue
        image_data = downloaded[hit.image_url]
        face = face_matcher.crop_face(query_encoding, image_data)
        to_analyse[hit.image_url] = face or image_data
    if not to_analyse:
        return
    deepfake_results = await deepfake_client.analyse_batch(to_analyse)

    for hit in hits:
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


async def _trace_deepfake(fake, downloaded: dict, known: set,
                          query_encoding, stats: dict) -> list:
    """
    Antistrofi anazhthsh me to idio to deepfake.

    Sygkrinoume ta apotelesmata me to proswpo tou deepfake kai oxi me thn
    eikona eisodou, giati psaxnoume antigrafa tou idiou deepfake.
    """
    image_data = downloaded[fake.image_url]
    faces = face_matcher._extract_faces(image_data)
    if not faces:
        return []
    # An exei polla proswpa, pairnoume afto pou moiazei me ton xrhsth
    fake_face = min(faces, key=lambda item: face_matcher.reference_distance(
        query_encoding, np.array(item["embedding"])))
    fake_encoding = np.array(fake_face["embedding"])

    try:
        hits = await reverse_search.search(image_data, "deepfake.jpg")
    except Exception as exc:
        logger.warning("Apotyxia anazhthshs gia to deepfake: %s", exc)
        return []
    stats["traced_deepfakes"] += 1

    hits = [hit for hit in hits if hit.image_url not in known]
    known.update(hit.image_url for hit in hits)
    for hit in hits:
        hit.traced_from = fake.image_url
    trace_downloaded = await reverse_search.download_images(hits)
    downloaded.update(trace_downloaded)
    stats["trace_results"] += len(trace_downloaded)

    traced = _filter_faces(hits, trace_downloaded, fake_encoding, stats)
    logger.info("To deepfake %s vrethike se %d akomh eikones",
               fake.image_url[:60], len(traced))
    return traced


def _filter_faces(hits: list, downloaded: dict, query_encoding,
                  stats: dict, min_side: int = 0,
                  closest: bool = False) -> list:
    """Kratame osa apotelesmata exoun to idio proswpo me to encoding."""
    confirmed = []
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

        # Mikres eikones (eikonidia, logotypa) ths sarwshs selidwn
        if min_side and min(hit.img_width or 0, hit.img_height or 0) < min_side:
            continue

        is_match, distance = face_matcher.compare_with_candidate(
            query_encoding, image_data, closest=closest
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
    return confirmed


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
