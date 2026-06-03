from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort

from .utils import blend_mask_overlay
from .yolo_seg import LetterboxResult, postprocess_semantic_mask, postprocess_yolov8_seg, letterbox


@dataclass
class ModelIO:
    input_name: str
    input_hw: Tuple[int, int]


class OnnxGrapeSegmenter:
    def __init__(
        self,
        model_path: Union[str, Path],
        providers: Optional[List[str]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到模型文件: {self.model_path}")

        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = True
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(self.model_path), sess_options=sess_options, providers=providers)
        self.io = self._infer_io()

    def _infer_io(self) -> ModelIO:
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            # YOLO usually has 1 image input
            input_name = inputs[0].name
        else:
            input_name = inputs[0].name

        shape = inputs[0].shape  # e.g. [1,3,640,640] or ['batch',3,'h','w']
        h = 640
        w = 640
        if isinstance(shape, (list, tuple)) and len(shape) == 4:
            maybe_h = shape[2]
            maybe_w = shape[3]
            if isinstance(maybe_h, int) and isinstance(maybe_w, int):
                h, w = maybe_h, maybe_w

        return ModelIO(input_name=input_name, input_hw=(h, w))

    def _preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, LetterboxResult]:
        in_h, in_w = self.io.input_hw
        lb = letterbox(img_bgr, (in_h, in_w))

        img = lb.image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float32) / 255.0
        chw = np.transpose(img_rgb, (2, 0, 1))
        x = np.expand_dims(chw, axis=0)
        return x, lb

    def predict(
        self,
        img_bgr: np.ndarray,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        semantic_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        orig_h, orig_w = img_bgr.shape[:2]
        x, lb = self._preprocess(img_bgr)

        ort_outs = self.session.run(None, {self.io.input_name: x})
        ort_outs = [np.asarray(o) for o in ort_outs]

        in_h, in_w = self.io.input_hw

        # Heuristic: semantic (single output with (1,1,H,W) or similar)
        if len(ort_outs) == 1 and ort_outs[0].ndim in (3, 4):
            mask_u8 = postprocess_semantic_mask(
                ort_outs[0],
                orig_hw=(orig_h, orig_w),
                input_hw=(in_h, in_w),
                letterbox_res=lb,
                threshold=semantic_threshold,
            )
            overlay = blend_mask_overlay(img_bgr, mask_u8, color_bgr=(0, 0, 255), alpha=0.45)
            return {
                "mode": "semantic",
                "mask_u8": mask_u8,
                "overlay_bgr": overlay,
                "detections": [],
            }

        # Otherwise try YOLOv8-seg instance format
        inst = postprocess_yolov8_seg(
            ort_outs,
            orig_hw=(orig_h, orig_w),
            input_hw=(in_h, in_w),
            letterbox_res=lb,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )

        # merge instance masks into a single semantic-ish mask for display
        merged = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for m in inst.masks:
            merged[m] = 255

        overlay = blend_mask_overlay(img_bgr, merged, color_bgr=(0, 0, 255), alpha=0.45)

        dets: List[Dict[str, Any]] = []
        for box, score, cls in zip(inst.boxes_xyxy, inst.scores, inst.class_ids):
            x1, y1, x2, y2 = [float(v) for v in box]
            dets.append({"box": [x1, y1, x2, y2], "score": float(score), "class_id": int(cls)})

        return {
            "mode": "instance",
            "mask_u8": merged,
            "overlay_bgr": overlay,
            "detections": dets,
        }
