from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort

from .utils import blend_mask_overlay, remove_small_connected_components
from .yolo_seg import (
    LetterboxResult,
    letterbox,
    postprocess_semantic_prob_and_mask,
    postprocess_yolov8_seg,
)


def _draw_box_and_label(img_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int, text: str) -> None:
    h, w = img_bgr.shape[:2]
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))

    font_scale = max(1.2, w / 800.0)
    thickness = max(2, int(font_scale * 1.2))
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), thickness)

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pad = int(8 * font_scale)
    label_y = max(y1, th + int(20 * font_scale))

    bg_pt1 = (x1, max(0, label_y - th - pad))
    bg_pt2 = (min(w - 1, x1 + tw + pad), min(h - 1, label_y + int(pad / 2)))
    cv2.rectangle(img_bgr, bg_pt1, bg_pt2, (0, 180, 0), -1)
    cv2.putText(
        img_bgr,
        text,
        (x1 + int(pad / 2), min(h - 1, label_y - int(pad / 4))),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


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
        min_area: int = 100000,
        min_w: int = 40,
        min_h: int = 40,
    ) -> Dict[str, Any]:
        orig_h, orig_w = img_bgr.shape[:2]
        x, lb = self._preprocess(img_bgr)

        ort_outs = self.session.run(None, {self.io.input_name: x})
        ort_outs = [np.asarray(o) for o in ort_outs]

        in_h, in_w = self.io.input_hw

        # Heuristic: semantic (single output with (1,1,H,W) or similar)
        if len(ort_outs) == 1 and ort_outs[0].ndim in (3, 4):
            prob_map, mask_u8 = postprocess_semantic_prob_and_mask(
                ort_outs[0],
                orig_hw=(orig_h, orig_w),
                input_hw=(in_h, in_w),
                letterbox_res=lb,
                threshold=semantic_threshold,
            )

            mask_u8 = remove_small_connected_components(mask_u8, min_area=min_area, min_w=min_w, min_h=min_h)
            overlay = blend_mask_overlay(img_bgr, mask_u8, color_bgr=(0, 0, 255), alpha=0.45)

            # 基于连通域画框 + 置信度（取该连通域内概率均值）
            dets: List[Dict[str, Any]] = []
            binary = (mask_u8 > 0).astype(np.uint8)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            comps = []
            for lab in range(1, num):
                area = int(stats[lab, cv2.CC_STAT_AREA])
                x = int(stats[lab, cv2.CC_STAT_LEFT])
                y = int(stats[lab, cv2.CC_STAT_TOP])
                w = int(stats[lab, cv2.CC_STAT_WIDTH])
                h = int(stats[lab, cv2.CC_STAT_HEIGHT])
                if area <= 0:
                    continue
                m = labels == lab
                conf = float(np.mean(prob_map[m])) if np.any(m) else 0.0
                comps.append((area, x, y, w, h, conf))

            comps.sort(key=lambda t: t[0], reverse=True)
            for area, x, y, w, h, conf in comps:
                x1, y1, x2, y2 = x, y, x + w, y + h
                _draw_box_and_label(overlay, x1, y1, x2, y2, f"Grape {conf:.2f}")
                dets.append({"box": [float(x1), float(y1), float(x2), float(y2)], "score": conf, "class_id": 0, "area": area})

            return {
                "mode": "semantic",
                "mask_u8": mask_u8,
                "overlay_bgr": overlay,
                "detections": dets,
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

        merged = remove_small_connected_components(merged, min_area=min_area, min_w=min_w, min_h=min_h)

        overlay = blend_mask_overlay(img_bgr, merged, color_bgr=(0, 0, 255), alpha=0.45)

        dets: List[Dict[str, Any]] = []
        for box, score, cls, m in zip(inst.boxes_xyxy, inst.scores, inst.class_ids, inst.masks):
            x1, y1, x2, y2 = [float(v) for v in box]
            bw = int(max(0.0, x2 - x1))
            bh = int(max(0.0, y2 - y1))
            area = int(np.count_nonzero(m))

            if min_area > 0 and area < min_area:
                continue
            if min_w > 0 and bw < min_w:
                continue
            if min_h > 0 and bh < min_h:
                continue

            dets.append({"box": [x1, y1, x2, y2], "score": float(score), "class_id": int(cls), "area": area})
            _draw_box_and_label(overlay, int(x1), int(y1), int(x2), int(y2), f"Grape {float(score):.2f}")

        return {
            "mode": "instance",
            "mask_u8": merged,
            "overlay_bgr": overlay,
            "detections": dets,
        }
