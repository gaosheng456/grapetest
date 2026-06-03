import base64
import json
import sys
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parent

image_arg = sys.argv[1] if len(sys.argv) >= 2 else "grape-example1.jpg"
min_area = int(sys.argv[2]) if len(sys.argv) >= 3 else 50
min_w = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
min_h = int(sys.argv[4]) if len(sys.argv) >= 5 else 0

image_path = (Path(image_arg) if Path(image_arg).is_absolute() else (root / image_arg)).resolve()
assert image_path.exists(), f"Missing image: {image_path}"

url = (
    "http://127.0.0.1:8000/api/predict"
    f"?conf=0.25&iou=0.45&threshold=0.5&min_area={min_area}&min_w={min_w}&min_h={min_h}"
)

auth_url = "http://127.0.0.1:8000/api/auth/login"
auth_body = json.dumps({"identifier": "grape", "password": "123"}).encode("utf-8")
auth_req = urllib.request.Request(auth_url, data=auth_body, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(auth_req, timeout=10) as resp:
    auth_data = json.loads(resp.read().decode("utf-8"))
token = auth_data.get("token")
assert token, "Login failed: missing token"

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
lines = [
    '--' + boundary,
    'Content-Disposition: form-data; name="file"; filename="0.png"',
    'Content-Type: image/png',
    '',
]
body = '\r\n'.join(lines).encode('utf-8') + b'\r\n' + image_path.read_bytes() + b'\r\n' + ('--' + boundary + '--\r\n').encode('utf-8')

req = urllib.request.Request(url, data=body, headers={
    'Content-Type': f'multipart/form-data; boundary={boundary}',
    'Authorization': f'Bearer {token}',
})
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.loads(resp.read().decode('utf-8'))

overlay_path = root / "overlay_png_base64.png"
mask_path = root / "mask_png_base64.png"
overlay_path.write_bytes(base64.b64decode(data["overlay_png_base64"]))
mask_path.write_bytes(base64.b64decode(data["mask_png_base64"]))

print("image", str(image_path))
print("min_area", min_area)
print("min_w", min_w)
print("min_h", min_h)
print("mode", data.get("mode"))
print("detections", data.get("detections"))
print("saved", str(overlay_path), str(mask_path))

try:
    import cv2
    import numpy as np

    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is not None:
        binary = (m > 0).astype(np.uint8)
        num, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA] if num > 1 else []
        print("mask_components", int(num - 1))
        print("mask_foreground_pixels", int(binary.sum()))
        if len(areas):
            print("mask_min_area", int(areas.min()), "mask_max_area", int(areas.max()))
except Exception as e:
    print("mask_stats_error", str(e))
