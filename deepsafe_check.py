"""
Aftonomo script gia dokimes ton deepfake detectors (DeepSafe), xoris reverse
search i face matching.

Xrisi:
    python deepsafe_check.py eikona.jpg
    python deepsafe_check.py fakelos_me_eikones
    python deepsafe_check.py fakelos --degrade --out apotelesmata.csv

Me to --degrade, kathe eikona stelnetai se plires morfi KAI se
ypovathmismenes ekdoxes (resize, JPEG compression, kai syndiasmos
"thumbnail-like"), gia na doume pws peftei h anixneysh.
"""

import argparse
import base64
import csv
import io
import sys
from pathlib import Path

import httpx
from PIL import Image

DEEPSAFE_URL = "http://localhost:8000"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODELS = ["npr_deepfakedetection", "universalfakedetect"]

RESIZE_SIZES = [512, 256, 128]
JPEG_QUALITIES = [90, 70, 50]


def collect_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(f for f in p.iterdir()
                                if f.suffix.lower() in EXTENSIONS))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"Den vrethike: {item}", file=sys.stderr)
    return paths


def to_bytes(img: Image.Image, fmt: str, quality: int | None = None) -> bytes:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, "JPEG", quality=quality)
    else:
        img.save(buf, "PNG")
    return buf.getvalue()


def resized(img: Image.Image, max_side: int) -> Image.Image:
    copy = img.copy()
    copy.thumbnail((max_side, max_side), Image.LANCZOS)
    return copy


def make_variants(path: Path, degrade: bool):
    """Epistrefei lista (onoma_variant, bytes, width, height)."""
    raw = path.read_bytes()
    img = Image.open(io.BytesIO(raw))
    variants = [("original", raw, img.width, img.height)]

    if not degrade:
        return variants

    # Resize mono (PNG, xoris apoleies) -> apomonwnei to resize
    for size in RESIZE_SIZES:
        small = resized(img, size)
        variants.append((f"resize_{size}", to_bytes(small, "PNG"),
                         small.width, small.height))

    # JPEG mono (idia analysi) -> apomonwnei to compression
    for q in JPEG_QUALITIES:
        variants.append((f"jpeg_q{q}", to_bytes(img, "JPEG", q),
                         img.width, img.height))

    # Syndiasmos: mikro + compressed, opws ena thumbnail
    thumb = resized(img, 150)
    variants.append(("thumb_150_q75", to_bytes(thumb, "JPEG", 75),
                     thumb.width, thumb.height))
    return variants


def query(client: httpx.Client, data: bytes) -> dict:
    payload = {
        "media_type": "image",
        "image_data": base64.b64encode(data).decode("ascii"),
        "ensemble_method": "voting",
    }
    response = client.post(f"{DEEPSAFE_URL}/predict", json=payload,
                           timeout=240.0)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="arxeia i fakeloi")
    parser.add_argument("--degrade", action="store_true",
                        help="dokimase kai ypovathmismenes ekdoxes")
    parser.add_argument("--out", default="deepsafe_results.csv")
    args = parser.parse_args()

    paths = collect_paths(args.inputs)
    if not paths:
        print("Den vrethikan eikones.")
        sys.exit(1)

    fieldnames = ["file", "variant", "width", "height", "bytes",
                  "verdict", "confidence"]
    for m in MODELS:
        fieldnames += [f"{m}_prob", f"{m}_pred"]
    fieldnames.append("error")

    with httpx.Client() as client, \
            open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for path in paths:
            for name, data, w, h in make_variants(path, args.degrade):
                row = {"file": path.name, "variant": name, "width": w,
                       "height": h, "bytes": len(data)}
                try:
                    result = query(client, data)
                    row["verdict"] = result.get("verdict")
                    row["confidence"] = result.get("confidence_in_verdict")
                    models = result.get("model_results", {})
                    for m in MODELS:
                        info = models.get(m, {})
                        row[f"{m}_prob"] = info.get("probability")
                        row[f"{m}_pred"] = info.get("class")
                except Exception as exc:  # noqa: BLE001
                    row["error"] = str(exc)

                writer.writerow(row)
                fh.flush()
                print(f"{path.name:25} {name:15} "
                      f"{row.get('verdict', 'ERROR'):6} "
                      f"npr={row.get('npr_deepfakedetection_prob')} "
                      f"ufd={row.get('universalfakedetect_prob')}")

    print(f"\nTo CSV apothikeftike sto {args.out}")


if __name__ == "__main__":
    main()
