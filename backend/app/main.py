from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .auth_routes import require_user, router as auth_router
from .auth_store import ensure_default_user
from .inference import OnnxGrapeSegmenter
from .utils import encode_png_base64, imdecode_image


def _default_model_path() -> Path:
    # repo_root/backend/app/main.py -> repo_root is parents[2]
    # app -> backend -> repo_root
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "yolo_seg.onnx"


def _resolve_model_path() -> Path:
    env_path = os.getenv("GRAPE_MODEL_PATH")
    if env_path:
        return Path(env_path)
    return _default_model_path()


app = FastAPI(title="Grape Segmentation API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

segmenter: Optional[OnnxGrapeSegmenter] = None


@app.on_event("startup")
def _startup() -> None:
    global segmenter
    ensure_default_user()
    model_path = _resolve_model_path()
    segmenter = OnnxGrapeSegmenter(model_path)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/api/model")
def model_info() -> Dict[str, Any]:
    if segmenter is None:
        raise RuntimeError("模型尚未加载")

    sess = segmenter.session
    return {
        "model_path": str(segmenter.model_path),
        "inputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in sess.get_inputs()],
        "outputs": [{"name": o.name, "shape": o.shape, "type": o.type} for o in sess.get_outputs()],
    }


@app.post("/api/predict")
async def predict(
    user: str = Depends(require_user),
    file: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.45,
    threshold: float = 0.5,
    min_area: int = 100000,
    min_w: int = 40,
    min_h: int = 40,
) -> Dict[str, Any]:
    if segmenter is None:
        raise RuntimeError("模型尚未加载")

    data = await file.read()
    img_bgr = imdecode_image(data)

    result = segmenter.predict(
        img_bgr,
        conf_thres=conf,
        iou_thres=iou,
        semantic_threshold=threshold,
        min_area=min_area,
        min_w=min_w,
        min_h=min_h,
    )

    mask_b64 = encode_png_base64(result["mask_u8"])
    overlay_b64 = encode_png_base64(result["overlay_bgr"])

    return {
        "mode": result["mode"],
        "mask_png_base64": mask_b64,
        "overlay_png_base64": overlay_b64,
        "detections": result["detections"],
        "filename": file.filename,
    }
