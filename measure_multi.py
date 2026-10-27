"""
Metrisi me perissoteres fwtografies anaforas (kef. 4.6).

Kathe fwtografia psaxnetai mia fora ana epanalipsi (2 klisis SerpApi).
Gia kathe apotelesma kratame ta embeddings twn proswpwn tou, wste meta na
ypologisoume xwris nees anazhthseis ti tha evriske h efarmogh me 1, 2, ..., 5
fwtografies: enwsi twn apotelesmatwn kai filtro proswpou me to meso oro.

Gia kathe plhthos k pairnoume oous tous syndyasmous k fwtografiwn kai
vgazoume to meso oro, wste na min exartatai apo to poies dialexame.

Dimiourgei:
  <out>_hits.csv     ena row ana apotelesma kai fwtografia
  <out>_faces.npz    ta embeddings twn apotelesmatwn
  <out>_summary.csv  ena row ana epanalipsi kai plhthos fwtografiwn

Xrisi (apo ton fakelo ptyxiakh, me energo to venv311):
    python measure_multi.py me_2.jpg me_5.jpg me_8.jpg me_10.jpg me_12.jpg --runs 2 --out multi
    python measure_multi.py ... --out multi --analyse-only
"""

import argparse
import asyncio
import csv
import logging
import re
import statistics
from itertools import combinations
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from app import config, face_matcher, reverse_search

TEST_SITE = "ankamim.github.io"
TEST_IMAGE = re.compile(r"/((?:me|df)_\d+)\.(?:jpe?g|png)", re.IGNORECASE)

HIT_FIELDS = ["run", "image", "engine", "match_type", "status", "faces",
              "test_image", "domain", "image_url", "source_url"]
SUMMARY_FIELDS = ["run", "photos", "combos", "found", "domains",
                  "matches_single_0.60", "matches_mean_0.50",
                  "match_domains", "test_images", "deepfakes",
                  "outside_test_site"]


def domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def test_image_of(hit: dict) -> str:
    if TEST_SITE not in hit["image_url"] + " " + hit["source_url"]:
        return ""
    match = TEST_IMAGE.search(hit["image_url"])
    return match.group(1).lower() if match else ""


async def search_all(paths: list[Path], runs: int, out: str) -> None:
    faces = {}
    with open(f"{out}_hits.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HIT_FIELDS)
        writer.writeheader()
        for run_no in range(1, runs + 1):
            for path in paths:
                hits = await reverse_search.search(path.read_bytes(), path.name)
                downloaded = await reverse_search.download_images(hits)
                for hit in hits:
                    data = downloaded.get(hit.image_url)
                    embeddings = []
                    if data is None:
                        status = "download_failed"
                    else:
                        if hit.image_url not in faces:
                            faces[hit.image_url] = face_matcher._extract_embedding(data)
                        embeddings = faces[hit.image_url]
                        status = "face" if embeddings else "no_face"
                    row = {
                        "run": run_no, "image": path.name, "engine": hit.engine,
                        "match_type": hit.match_type, "status": status,
                        "faces": len(embeddings), "domain": domain(hit.source_url),
                        "image_url": hit.image_url, "source_url": hit.source_url,
                    }
                    row["test_image"] = test_image_of(row)
                    writer.writerow(row)
                fh.flush()
                print(f"run {run_no} {path.name:12} found={len(hits)} "
                      f"downloaded={len(downloaded)}")

    keys = sorted(faces)
    np.savez(f"{out}_faces.npz", urls=np.array(keys, dtype=object),
             faces=np.array([np.array(faces[k]) for k in keys], dtype=object))


def analyse(paths: list[Path], out: str) -> None:
    refs = {p.name: face_matcher.encode_query_face(p.read_bytes()) for p in paths}
    stored = np.load(f"{out}_faces.npz", allow_pickle=True)
    faces = dict(zip(stored["urls"], stored["faces"]))

    with open(f"{out}_hits.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_run: dict[str, list[dict]] = {}
    for row in rows:
        by_run.setdefault(row["run"], []).append(row)

    def best(url: str, names: tuple) -> float | None:
        embeddings = faces.get(url)
        if embeddings is None or len(embeddings) == 0:
            return None
        encodings = [refs[n] for n in names]
        return min(face_matcher.reference_distance(encodings, e)
                   for e in embeddings)

    summary = []
    for run, run_rows in sorted(by_run.items()):
        for k in range(1, len(paths) + 1):
            per_combo = []
            for names in combinations(refs, k):
                hits = {}
                for row in run_rows:
                    if row["image"] in names:
                        hits.setdefault(row["image_url"], row)
                threshold = (config.FACE_MULTI_REF_THRESHOLD if k > 1
                             else config.FACE_DISTANCE_THRESHOLD)
                matches = [r for url, r in hits.items()
                           if (d := best(url, names)) is not None and d < threshold]
                # Kanonas mias fwtografias panw sta idia apotelesmata, gia
                # sygkrish: arkei na moiazei se mia apo tis fwtografies.
                single = [url for url in hits
                          if any((d := best(url, (n,))) is not None
                                 and d < config.FACE_DISTANCE_THRESHOLD
                                 for n in names)]
                own = {Path(n).stem.lower() for n in names}
                tests = {r["test_image"] for r in matches} - own - {""}
                per_combo.append({
                    "found": len(hits),
                    "domains": len({r["domain"] for r in hits.values()}),
                    "matches_single_0.60": len(single),
                    "matches_mean_0.50": len(matches),
                    "match_domains": len({r["domain"] for r in matches}),
                    "test_images": len(tests),
                    "deepfakes": len({t for t in tests if t.startswith("df_")}),
                    "outside_test_site": sum(1 for r in matches
                                             if TEST_SITE not in r["image_url"] + r["source_url"]),
                })
            row = {"run": run, "photos": k, "combos": len(per_combo)}
            for field in SUMMARY_FIELDS[3:]:
                row[field] = round(statistics.mean(c[field] for c in per_combo), 2)
            summary.append(row)
            print(" ".join(f"{key}={value}" for key, value in row.items()))

    with open(f"{out}_summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nTa apotelesmata apothikeftikan sto {out}_summary.csv")


async def main(args) -> None:
    paths = [Path(p) for p in args.images][:config.MAX_QUERY_IMAGES]
    if not args.analyse_only:
        await search_all(paths, args.runs, args.out)
    analyse(paths, args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="oi fwtografies anaforas (mexri 5)")
    parser.add_argument("--runs", type=int, default=1,
                        help="epanalipseis (kathe mia kostizei 2 klisis ana fwtografia)")
    parser.add_argument("--out", default="multi", help="prothema arxeion")
    parser.add_argument("--analyse-only", action="store_true",
                        help="mono analysh panw se proigoumena apotelesmata")
    cli = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    asyncio.run(main(cli))
