from __future__ import annotations

import base64
from typing import Tuple

import cv2
import numpy as np


def imdecode_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图片：请确认上传的是 png/jpg")
    return img


def encode_png_base64(img_bgr_or_gray: np.ndarray) -> str:
    if img_bgr_or_gray.ndim == 2:
        ok, buf = cv2.imencode(".png", img_bgr_or_gray)
    else:
        ok, buf = cv2.imencode(".png", img_bgr_or_gray)
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def blend_mask_overlay(image_bgr: np.ndarray, mask_u8: np.ndarray, color_bgr: Tuple[int, int, int] = (0, 0, 255), alpha: float = 0.45) -> np.ndarray:
    if mask_u8.ndim != 2:
        raise ValueError("mask 必须是单通道")
    overlay = image_bgr.copy()
    colored = np.zeros_like(image_bgr)
    colored[:, :] = np.array(color_bgr, dtype=np.uint8)
    m = mask_u8.astype(bool)
    if np.any(m):
        base = image_bgr.astype(np.float32)
        col = colored.astype(np.float32)
        overlay_f = overlay.astype(np.float32)
        overlay_f[m] = base[m] * (1.0 - alpha) + col[m] * alpha
        overlay = np.clip(overlay_f, 0, 255).astype(np.uint8)
    return overlay
