"""
Epalitheusi prosopou.

Ta apotelesmata tou reverse search einai "optika omoies eikones", den periexoun 
anagkastika to idio prosopo. Edw filtraroume kai kratame mono
osa apotelesmata periexoun proswpo pou tairiazei me thn eikona tou xrhsth.
"""

import io
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# To DeepFace typwnei emoji molis ginei import. Sta Windows, otan to output
# paei se arxeio, to cp1252 den ta kserei kai to programma skaei.
sys.stdout.reconfigure(errors="replace")

from deepface import DeepFace

from . import config

logger = logging.getLogger(__name__)

# To VGG-Face den xexwrize kala ta idia proswpa apo ksenous kai petouse
# ta perissotera deepfakes prin ftasoun sto DeepSafe.
MODEL_NAME = "Facenet512"

DETECTOR_BACKEND = "mtcnn"


class NoFaceFoundError(Exception):
    """Den vrethhke proswpo sthn eikona."""


class MultipleFacesError(Exception):
    """Vrethhkan polla proswpa - H eikona eisodou prepei na exei mono ena."""


def _load_and_resize(image_bytes: bytes) -> Image.Image:
    """
    Diavazei bytes se PIL Image, me smikrinsi gia taxytita.

    Metatrepoume se RGB wste na doulevei asxetws tou arxikou format.
    """
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")

    if max(image.size) > config.MAX_IMAGE_DIMENSION:
        ratio = config.MAX_IMAGE_DIMENSION / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    return image


def _save_temp(image: Image.Image) -> str:
    """
    To DeepFace pairnei pio axiopista path arxeiou para numpy array
    apo tin mnhmh (kalyterh symvatothta anamesa se ekdoseis). Apothikeuoume
    prosorina se ena arxeio kai to diagrafoume meta.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    image.save(tmp.name, format="JPEG")
    return tmp.name


def _extract_faces(image_bytes: bytes) -> list[dict]:
    """
    Vgazei ola ta prosopa pou vriskei to DeepFace sthn eikona, mazi me to
    embedding kai th thesi tous (facial_area). Epistrefei empty list an den
    vrei tipota.

    Xrhsimopoioume enforce_detection=True wste to idio to DeepFace na
    kanei ton elegxo kai na petaxei sfalma an den vrei proswpo,
    anti na vasizomaste sto pedio face_confidence pou einai poio anaksiopisto 
    giati symperiferetai diaforetika analoga me thn ekdosh.
    """
    image = _load_and_resize(image_bytes)
    path = _save_temp(image)

    try:
        results = DeepFace.represent(
            img_path=path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )
    except ValueError as exc:
        # Afto to petaei to DeepFace otan den entopIzei proswpo.
        logger.info("Den entopisthke proswpo: %s", exc)
        return []
    except Exception as exc:
        logger.warning("Aprosmeno sfalma sto DeepFace: %s", exc)
        return []
    finally:
        Path(path).unlink(missing_ok=True)

    return results


def _extract_embedding(image_bytes: bytes) -> list[np.ndarray]:
    """
    Vgazei ta embeddings olwn twn prosopwn pou vriskei to DeepFace
    sthn eikona. Epistrefei empty list an den vrei tipota.
    """
    return [np.array(item["embedding"]) for item in _extract_faces(image_bytes)]


def encode_query_face(image_bytes: bytes) -> np.ndarray:
    """
    Vgazei to embedding apo thn eikona pou anevase o xrhsths.

    H eikona anaforas prepei na exei akrivos ena kathara orato prosopo, 
    alliws den kseroume poion psaxnoume.
    """
    embeddings = _extract_embedding(image_bytes)

    if len(embeddings) == 0:
        raise NoFaceFoundError(
            "Den entopisthke proswpo sthn eikona. "
            "Dokimase mia katharh fwtografia me orato proswpo."
        )
    if len(embeddings) > 1:
        raise MultipleFacesError(
            f"Entopisthkan {len(embeddings)} proswpa. "
            "Anevase fwtografia me ena mono proswpo."
        )

    return embeddings[0]


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Apostash synhmhnitonou anamesa se dyo embeddings.
    0 simainei taftosima, 1 simainei entelws diaforetika.
    """
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    similarity = float(np.dot(a_norm, b_norm))
    return 1.0 - similarity


def _as_refs(query_encoding) -> list[np.ndarray]:
    """Ena encoding h lista apo encodings (mia ana fwtografia anaforas)."""
    if isinstance(query_encoding, np.ndarray) and query_encoding.ndim == 1:
        return [query_encoding]
    return list(query_encoding)


def reference_distance(query_encoding, embedding: np.ndarray) -> float:
    """
    Apostash enos proswpou apo tis fwtografies anaforas.

    Me mia fwtografia einai h apostash ths. Me perissoteres pairnoume to
    meso oro, giati enas ksenos pou moiazei se mia fwtografia spania
    moiazei kai stis alles. Me 4 fwtografies kai katwfli 0.50 ta lookalikes
    tou Yandex epesan apo 15/18 se 7/18 kai pernousan akoma ola ta deepfakes.
    """
    refs = _as_refs(query_encoding)
    return float(np.mean([_cosine_distance(ref, embedding) for ref in refs]))


def match_threshold(query_encoding) -> float:
    """To katwfli analoga me to plhthos twn fwtografiwn anaforas."""
    if len(_as_refs(query_encoding)) > 1:
        return config.FACE_MULTI_REF_THRESHOLD
    return config.FACE_DISTANCE_THRESHOLD


def compare_with_candidate(query_encoding,
                           candidate_bytes: bytes) -> tuple[bool, float | None]:
    """
    Sygkrinei tis eikones anaforas me ena apotelesma ths anazhtishs.

    Epistrefei (einai_match, apostasi).
    An h ypopsifia eikona den exei proswwpo, epistrefei (False, None).

    Simeiosh: mia eikona apo to internet mporei na exei polla proswpa
    (px omadiki fwtografia). Kratame thn mikroteri apostasi, diladh to
    proswpo pou moiazei perissotero.
    """
    embeddings = _extract_embedding(candidate_bytes)
    if len(embeddings) == 0:
        return False, None

    distances = [reference_distance(query_encoding, e) for e in embeddings]
    best_distance = min(distances)
    is_match = best_distance < match_threshold(query_encoding)

    return is_match, best_distance


def crop_face(query_encoding,
              candidate_bytes: bytes) -> bytes | None:
    """
    Kovei to proswpo pou moiazei perissotero me thn eikona anaforas,
    gia na to steiloume sto DeepSafe anti gia olokliri thn eikona.

    To universalfakedetect kovei mono ena 224x224 kommati apo to kentro,
    opote se megales fwtografies den evlepe kan to proswpo. Me perithwrio
    30% kai megethos 256x256 epiase kai ta 4 deepfakes tou test set.
    An den vrethei proswpo, epistrefei None.
    """
    faces = _extract_faces(candidate_bytes)
    if len(faces) == 0:
        return None

    best = min(faces, key=lambda item: reference_distance(
        query_encoding, np.array(item["embedding"])))
    area = best["facial_area"]

    # To facial_area einai se syntetagmenes ths smikrymenis eikonas
    image = _load_and_resize(candidate_bytes)
    side = max(area["w"], area["h"]) * (1 + config.FACE_CROP_MARGIN)
    center_x = area["x"] + area["w"] / 2
    center_y = area["y"] + area["h"] / 2
    face = image.crop((
        int(center_x - side / 2), int(center_y - side / 2),
        int(center_x + side / 2), int(center_y + side / 2),
    ))
    size = config.FACE_CROP_SIZE
    face = face.resize((size, size), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    face.save(buffer, format="PNG")
    return buffer.getvalue()


def distance_to_similarity(distance: float) -> float:
    """
    Metatrepei thn apostash se pososto omoiotitas gia thn anafora.
    """
    similarity = max(0.0, 1.0 - distance)
    return round(similarity * 100, 1)