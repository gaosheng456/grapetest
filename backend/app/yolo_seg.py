from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class LetterboxResult:
    image: np.ndarray
    ratio: float
    pad: Tuple[int, int]  # (pad_x, pad_y)
    new_unpad: Tuple[int, int]  # (w, h) scaled image size without pad


def letterbox(
    img_bgr: np.ndarray,
    new_shape: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114),
) -> LetterboxResult:
    h0, w0 = img_bgr.shape[:2]
    new_h, new_w = new_shape

    r = min(new_w / w0, new_h / h0)
    scaled_w, scaled_h = int(round(w0 * r)), int(round(h0 * r))

    resized = cv2.resize(img_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_w - scaled_w
    pad_h = new_h - scaled_h
    pad_x = pad_w // 2
    pad_y = pad_h // 2

    out = cv2.copyMakeBorder(
        resized,
        top=pad_y,
        bottom=pad_h - pad_y,
        left=pad_x,
        right=pad_w - pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )

    return LetterboxResult(image=out, ratio=r, pad=(pad_x, pad_y), new_unpad=(scaled_w, scaled_h))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
    out = xywh.copy()
    out[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    out[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    out[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    out[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return out


def clip_boxes_xyxy(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes


def box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    xA = np.maximum(box[0], boxes[:, 0])
    yA = np.maximum(box[1], boxes[:, 1])
    xB = np.minimum(box[2], boxes[:, 2])
    yB = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, xB - xA)
    inter_h = np.maximum(0.0, yB - yA)
    inter = inter_w * inter_h

    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-9
    return inter / union


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    idxs = scores.argsort()[::-1]
    keep: List[int] = []
    while idxs.size > 0:
        i = int(idxs[0])
        keep.append(i)
        if idxs.size == 1:
            break
        ious = box_iou_xyxy(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return keep


@dataclass
class InstanceSegResult:
    boxes_xyxy: np.ndarray  # (n,4) in original image coords
    scores: np.ndarray  # (n,)
    class_ids: np.ndarray  # (n,)
    masks: List[np.ndarray]  # list of (H,W) bool masks in original image coords


def postprocess_yolov8_seg(
    outputs: List[np.ndarray],
    orig_hw: Tuple[int, int],
    input_hw: Tuple[int, int],
    letterbox_res: LetterboxResult,
    conf_thres: float,
    iou_thres: float,
) -> InstanceSegResult:
    # Expect 2 outputs: pred (1, n, 4+nc+nm) and proto (1, nm, mh, mw)
    if len(outputs) < 2:
        raise ValueError("该模型输出不符合常见 YOLOv8-seg (pred+proto) 格式")

    pred = None
    proto = None
    for out in outputs:
        if out.ndim == 3:
            pred = out
        elif out.ndim == 4:
            proto = out

    if pred is None or proto is None:
        raise ValueError("无法从 ONNX 输出中识别 pred/proto")

    pred = np.squeeze(pred, axis=0)  # (n, d)
    proto = np.squeeze(proto, axis=0)  # (nm, mh, mw)

    # Some ONNX exports produce pred as (d, n) after squeeze (i.e. original was (1, d, n)).
    # Detect this by leveraging proto channel count (nm).
    if pred.ndim == 2:
        nm, _, _ = proto.shape
        # If first dim looks like feature dim (4+nc+nm) and second dim is large, transpose.
        if pred.shape[0] <= pred.shape[1] and pred.shape[0] >= 4 + nm:
            pred = pred.T

    nm, mh, mw = proto.shape
    d = pred.shape[1]
    if d <= 4 + nm:
        raise ValueError(f"pred 维度异常: {pred.shape}, proto={proto.shape}")

    nc = d - 4 - nm
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4 : 4 + nc]
    mask_coeffs = pred[:, 4 + nc :]

    if nc == 1:
        scores = cls_scores[:, 0]
        class_ids = np.zeros_like(scores, dtype=np.int64)
    else:
        class_ids = np.argmax(cls_scores, axis=1).astype(np.int64)
        scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]

    keep = scores >= conf_thres
    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    mask_coeffs = mask_coeffs[keep]

    if boxes_xywh.shape[0] == 0:
        return InstanceSegResult(
            boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            class_ids=np.zeros((0,), dtype=np.int64),
            masks=[],
        )

    boxes_xyxy = xywh2xyxy(boxes_xywh)

    in_h, in_w = input_hw
    boxes_xyxy = clip_boxes_xyxy(boxes_xyxy, in_w, in_h)

    keep_idx = nms_xyxy(boxes_xyxy, scores, iou_thres)
    boxes_xyxy = boxes_xyxy[keep_idx]
    scores = scores[keep_idx]
    class_ids = class_ids[keep_idx]
    mask_coeffs = mask_coeffs[keep_idx]

    # masks: (n, mh, mw)
    proto_flat = proto.reshape(nm, -1)  # (nm, mh*mw)
    masks = sigmoid(mask_coeffs @ proto_flat).reshape(-1, mh, mw)

    # upsample to model input size
    masks_up: List[np.ndarray] = []
    for m in masks:
        m_up = cv2.resize(m, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        masks_up.append(m_up)

    # remove letterbox padding and resize back to original size
    pad_x, pad_y = letterbox_res.pad
    scaled_w, scaled_h = letterbox_res.new_unpad
    orig_h, orig_w = orig_hw

    final_masks: List[np.ndarray] = []
    for m_up, box_in in zip(masks_up, boxes_xyxy):
        # crop to unpadded area first
        m_unpad = m_up[pad_y : pad_y + scaled_h, pad_x : pad_x + scaled_w]
        m_orig = cv2.resize(m_unpad, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        m_bin = m_orig > 0.5

        # Optional: crop by box (now in orig coords)
        final_masks.append(m_bin)

    # scale boxes to original coords
    # box_in is in input coords; undo pad then divide by ratio
    boxes_orig = boxes_xyxy.copy().astype(np.float32)
    boxes_orig[:, [0, 2]] -= pad_x
    boxes_orig[:, [1, 3]] -= pad_y
    boxes_orig /= letterbox_res.ratio
    boxes_orig = clip_boxes_xyxy(boxes_orig, orig_w, orig_h)

    return InstanceSegResult(
        boxes_xyxy=boxes_orig.astype(np.float32),
        scores=scores.astype(np.float32),
        class_ids=class_ids.astype(np.int64),
        masks=final_masks,
    )


def postprocess_semantic_mask(
    output: np.ndarray,
    orig_hw: Tuple[int, int],
    input_hw: Tuple[int, int],
    letterbox_res: LetterboxResult,
    threshold: float = 0.5,
) -> np.ndarray:
    # Accept (1,1,H,W) or (1,H,W) or (H,W)
    if output.ndim == 4:
        out = output[0]
        if out.shape[0] == 1:
            out = out[0]
        else:
            # multi-class: take argmax
            out = np.argmax(out, axis=0).astype(np.float32)
    elif output.ndim == 3:
        out = output[0]
    else:
        out = output

    out = out.astype(np.float32)
    # If values look like logits, apply sigmoid
    if out.max() > 1.0 or out.min() < 0.0:
        out = sigmoid(out)

    in_h, in_w = input_hw
    out_up = cv2.resize(out, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    pad_x, pad_y = letterbox_res.pad
    scaled_w, scaled_h = letterbox_res.new_unpad
    orig_h, orig_w = orig_hw

    out_unpad = out_up[pad_y : pad_y + scaled_h, pad_x : pad_x + scaled_w]
    out_orig = cv2.resize(out_unpad, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask = (out_orig >= threshold).astype(np.uint8) * 255
    return mask


def postprocess_semantic_prob_and_mask(
    output: np.ndarray,
    orig_hw: Tuple[int, int],
    input_hw: Tuple[int, int],
    letterbox_res: LetterboxResult,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (prob_map, mask_u8)。

    - prob_map: (H,W) float32，范围约为 [0,1]
    - mask_u8: (H,W) uint8，0/255
    """
    # 复用原逻辑，但保留 out_orig 概率图
    if output.ndim == 4:
        out = output[0]
        if out.shape[0] == 1:
            out = out[0]
        else:
            out = np.argmax(out, axis=0).astype(np.float32)
    elif output.ndim == 3:
        out = output[0]
    else:
        out = output

    out = out.astype(np.float32)
    if out.max() > 1.0 or out.min() < 0.0:
        out = sigmoid(out)

    in_h, in_w = input_hw
    out_up = cv2.resize(out, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    pad_x, pad_y = letterbox_res.pad
    scaled_w, scaled_h = letterbox_res.new_unpad
    orig_h, orig_w = orig_hw

    out_unpad = out_up[pad_y : pad_y + scaled_h, pad_x : pad_x + scaled_w]
    out_orig = cv2.resize(out_unpad, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    mask = (out_orig >= threshold).astype(np.uint8) * 255
    return out_orig, mask
