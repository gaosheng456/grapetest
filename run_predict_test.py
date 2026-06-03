import base64
import json
import urllib.request
from pathlib import Path

url = 'http://127.0.0.1:8000/api/predict?conf=0.25&iou=0.45&threshold=0.5'
image_path = Path('0.png')
assert image_path.exists(), f'Missing sample image: {image_path}'

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
lines = [
    '--' + boundary,
    'Content-Disposition: form-data; name="file"; filename="0.png"',
    'Content-Type: image/png',
    '',
]
body = '\r\n'.join(lines).encode('utf-8') + b'\r\n' + image_path.read_bytes() + b'\r\n' + ('--' + boundary + '--\r\n').encode('utf-8')

req = urllib.request.Request(url, data=body, headers={
    'Content-Type': f'multipart/form-data; boundary={boundary}'
})
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.loads(resp.read().decode('utf-8'))

Path('overlay_png_base64.png').write_bytes(base64.b64decode(data['overlay_png_base64']))
Path('mask_png_base64.png').write_bytes(base64.b64decode(data['mask_png_base64']))
print('mode', data['mode'])
print('detections', data.get('detections'))
print('saved overlay_png_base64.png mask_png_base64.png')
