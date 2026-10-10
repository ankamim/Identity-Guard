"""
Metrisi tis anazitisis (kef. 4.3).

Gia kathe eikona kai kathe epanalipsi trexei: anazitisi -> lipsi eikonon ->
apostasi prosopou gia KATHE apotelesma. Den trexei elegxos allioisis (DeepSafe),
opote einai grigoro kai den xreiazetai to Docker.

Dimiourgei dyo arxeia CSV:
  <out>_hits.csv  ena row ana apotelesma (me apostasi prosopou)
  <out>_runs.csv  ena row ana epanalipsi (metrites, recall, xronoi)

To recall metraei poses apo tis ALLES eikones tou peiramatos (me_*, df_*)
vrethikan, vasei tou URL tous sto GitHub Pages. An i mixani epistrepsei
mikrografia (thumbnail), to onoma tis eikonas den fainetai sto URL kai den
metraei. Gi' afto to recall einai katotato orio.

Xrisi (apo ton fakelo ptyxiakh, me energo to venv311):
    python measure_search.py "C:/Users/kamil/Pictures/thesis/me" --runs 1 --out search
    python measure_search.py "C:/.../me" --runs 3 --out search --public-base https://ankamim.github.io/deepfake_detector_test_images/

Prosoxi: kathe epanalipsi kanei 2 klisis SerpApi (3 an dwtheis --public-base).
Kathe epanalipsi paei ~2-4 lepta (i sygkrisi prosopon ginetai seiriaka).
"""

import argparse
import asyncio
import csv
import logging
import re
import sys
import time
from pathlib import Path

from app import face_matcher, reverse_search

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
THRESHOLDS = (0.50, 0.55, 0.60, 0.65)
RECALL_THRESHOLD = 0.60
TEST_SITE = "ankamim.github.io"
TEST_IMAGE = re.compile(r"/((?:me|df)_\d+)\.(?:jpe?g|png)", re.IGNORECASE)

HIT_FIELDS = ["image", "run", "engine", "match_type", "status", "distance",
              "in_test_site", "test_image", "image_url", "source_url"]
RUN_FIELDS = (["image", "run", "public_url_used", "found", "downloaded",
               "with_face"]
              + [f"matches_{t:.2f}" for t in THRESHOLDS]
              + ["self_found", "others_found", "others_list",
                 "sec_search", "sec_download", "sec_faces"])


def collect(path: str) -> list[Path]:
    p = Path(path)
    if p.is_dir():
        return sorted(f for f in p.iterdir()
                      if f.suffix.lower() in EXTENSIONS)
    return [p]


def open_csv(path: str, fields: list[str]):
    exists = Path(path).exists() and Path(path).stat().st_size > 0
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fields)
    if not exists:
        writer.writeheader()
    return fh, writer


async def run_once(path: Path, run_no: int, public_base: str | None,
                   hits_w, runs_w, hits_fh, runs_fh) -> None:
    name = path.name
    stem = path.stem.lower()
    image_bytes = path.read_bytes()

    try:
        query = face_matcher.encode_query_face(image_bytes)
    except Exception as exc:  # noqa: BLE001
        print(f"{name} run {run_no}: apotyxia encoding ({exc})")
        return

    public_url = (public_base.rstrip("/") + "/" + name) if public_base else None

    t0 = time.time()
    hits = await reverse_search.search(image_bytes, name, public_url)
    sec_search = time.time() - t0

    t0 = time.time()
    downloaded = await reverse_search.download_images(hits)
    sec_download = time.time() - t0

    t0 = time.time()
    distances: list[float] = []
    near_test_images: dict[str, float] = {}

    for hit in hits:
        data = downloaded.get(hit.image_url)
        dist = None
        if data is None:
            status = "download_failed"
        else:
            _, dist = face_matcher.compare_with_candidate(query, data)
            status = "no_face" if dist is None else "face"
            if dist is not None:
                distances.append(dist)

        in_site = TEST_SITE in (hit.image_url + " " + hit.source_url)
        match = TEST_IMAGE.search(hit.image_url) if in_site else None
        test_image = match.group(1).lower() if match else ""

        if test_image and dist is not None and dist < RECALL_THRESHOLD:
            near_test_images[test_image] = min(
                dist, near_test_images.get(test_image, 1.0))

        hits_w.writerow({
            "image": name, "run": run_no, "engine": hit.engine,
            "match_type": getattr(hit, "match_type", ""),
            "status": status,
            "distance": "" if dist is None else round(dist, 4),
            "in_test_site": in_site, "test_image": test_image,
            "image_url": hit.image_url, "source_url": hit.source_url,
        })
    hits_fh.flush()
    sec_faces = time.time() - t0

    others = sorted(i for i in near_test_images if i != stem)
    row = {
        "image": name, "run": run_no,
        "public_url_used": bool(public_url),
        "found": len(hits), "downloaded": len(downloaded),
        "with_face": len(distances),
        "self_found": stem in near_test_images,
        "others_found": len(others), "others_list": " ".join(others),
        "sec_search": round(sec_search, 1),
        "sec_download": round(sec_download, 1),
        "sec_faces": round(sec_faces, 1),
    }
    for t in THRESHOLDS:
        row[f"matches_{t:.2f}"] = sum(d < t for d in distances)
    runs_w.writerow(row)
    runs_fh.flush()

    print(f"{name:14} run {run_no}: found={len(hits)} "
          f"downloaded={len(downloaded)} face={len(distances)} "
          f"match@0.60={row['matches_0.60']} others={len(others)} "
          f"({sec_search + sec_download + sec_faces:.0f} s)")


async def main(args) -> None:
    images = collect(args.images)
    if args.limit:
        images = images[:args.limit]
    if not images:
        print("Den vrethikan eikones.")
        sys.exit(1)

    hits_fh, hits_w = open_csv(f"{args.out}_hits.csv", HIT_FIELDS)
    runs_fh, runs_w = open_csv(f"{args.out}_runs.csv", RUN_FIELDS)
    try:
        for path in images:
            for run_no in range(1, args.runs + 1):
                await run_once(path, run_no, args.public_base,
                               hits_w, runs_w, hits_fh, runs_fh)
    finally:
        hits_fh.close()
        runs_fh.close()
    print(f"\nTa apotelesmata apothikeftikan sta {args.out}_hits.csv "
          f"kai {args.out}_runs.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", help="arxeio i fakelos me eikones")
    parser.add_argument("--runs", type=int, default=1,
                        help="epanalipseis ana eikona")
    parser.add_argument("--out", default="search",
                        help="prothema arxeion CSV")
    parser.add_argument("--public-base", default=None,
                        help="dimosio base URL (px GitHub Pages) gia to Yandex")
    parser.add_argument("--limit", type=int, default=0,
                        help="mono oi prwtes N eikones (gia dokimi)")
    cli = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    asyncio.run(main(cli))
