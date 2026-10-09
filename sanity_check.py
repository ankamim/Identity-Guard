"""
Aftonomo script gia sanity check tou DeepSafe.

Stelnei mia mono eikona apeftheias sto DeepSafe /predict, xoris na
perasei apo reverse search i face matching. Xrisimo gia na epivevaioseis
an to montelo dvoulevei swsta se gnosta deigmata, anexartita apo to
ypoloipo pipeline.

Xrisi:
    python sanity_check.py path\pros\eikona.jpg
"""

import base64
import sys
 
import httpx
 
DEEPSAFE_URL = "http://localhost:8000"
 
 
def check_image(path: str) -> None:
    with open(path, "rb") as f:
        image_bytes = f.read()
 
    encoded = base64.b64encode(image_bytes).decode("ascii")
 
    payload = {
        "media_type": "image",
        "image_data": encoded,
        "ensemble_method": "voting",
    }
 
    print(f"Stelno {path} ({len(image_bytes)} bytes) sto DeepSafe...")
    print("(I proti klisi mporei na paei arga - to universalfakedetect")
    print(" fortonei ena CLIP montelo tin proti fora, mexri 3-4 lepta.)")
    response = httpx.post(f"{DEEPSAFE_URL}/predict", json=payload, timeout=240.0)
    response.raise_for_status()
    result = response.json()
 
    print("\n--- Apotelesma ---")
    print(f"Verdict:    {result.get('verdict')}")
    print(f"Confidence: {result.get('confidence_in_verdict')}")
    print(f"Raw:        {result.get('model_results')}")
 
 
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Xrisi: python sanity_check.py path\\pros\\eikona.jpg")
        sys.exit(1)
 
    check_image(sys.argv[1])