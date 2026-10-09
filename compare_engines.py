"""
Sygkrisi Google Lens kai Yandex gia mia eikona pou einai idi dimosia.

To Yandex dexetai mono dimosio URL. Gia ta peiramata xrisimopoioume tis
eikones pou filoxenoume sto GitHub Pages, opote den xreiazetai tunnel.

Xrisi (apo ton fakelo ptyxiakh, me energo to venv311):
    python compare_engines.py me_12.jpg https://ankamim.github.io/deepfake_detector_test_images/me_12.jpg

To proto orisma einai to topiko arxeio, to deutero to dimosio URL tis idias eikonas.
To DeepSafe prepei na trexei gia na ginei kai o elegxos allioisis.
"""

import asyncio
import logging
import sys
from collections import Counter
from pathlib import Path

from app import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)


async def main(path: str, public_url: str) -> None:
    image_bytes = Path(path).read_bytes()
    report = await pipeline.run(image_bytes, Path(path).name, public_url)

    by_engine = Counter(f["engine"] for f in report["findings"])

    print("\n--- Apotelesma ---")
    print("Statistika:", report["statistics"])
    print("Epivevaiomena eurimata ana mixani:", dict(by_engine))
    print("Diarkeia:", report["duration_seconds"], "sec")
    print("\nSimeiosi: ta diplotypa vasei URL krataei to proto (Lens), ")
    print("ara o arithmos tou Yandex einai katotatos.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Xrisi: python compare_engines.py <topiko_arxeio> <dimosio_url>")
        sys.exit(1)

    asyncio.run(main(sys.argv[1], sys.argv[2]))
