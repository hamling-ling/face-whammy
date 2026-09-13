#!/usr/bin/env python3
"""MediaPipe BlazeFace + Face Mesh via TFLite / LiteRT (no MediaPipe runtime)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DET_MODEL = HERE / "models" / "face_detection_short_range.tflite"
LM_MODEL = HERE / "models" / "face_landmark.tflite"
INT8_DET_MODEL = HERE / "models" / "int8" / "face_detection_front_128_integer_quant.tflite"
INT8_LM_MODEL = HERE / "models" / "int8" / "face_landmark_192_integer_quant.tflite"

QUANT_MODELS = {
    "fp32": (DET_MODEL, LM_MODEL),
    "int8": (INT8_DET_MODEL, INT8_LM_MODEL),
}

DET_SIZE = 128
LM_SIZE = 192
NUM_LANDMARKS = 468
MIN_SCORE = 0.5
NMS_THRESH = 0.3
FACE_FLAG_THRESH = 0.5
ROI_SCALE = 1.5
RAW_SCORE_LIMIT = 80.0
NUM_THREADS = 4


def make_interpreter(model_path: Path, num_threads: int = NUM_THREADS):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    try:
        interp = Interpreter(model_path=str(model_path), num_threads=num_threads)
    except TypeError:
        interp = Interpreter(model_path=str(model_path))
        setter = getattr(interp, "set_num_threads", None)
        if setter is not None:
            setter(num_threads)
    interp.allocate_tensors()
    return interp


def _ssd_anchors() -> np.ndarray:
    # face_detection_short_range: strides 8,16,16,16; interpolated_scale_aspect_ratio=1
    strides = (8, 16, 16, 16)
    anchors: list[tuple[float, float]] = []
    layer = 0
    while layer < len(strides):
        last = layer
        repeats = 0
        while last < len(strides) and strides[last] == strides[layer]:
            last += 1
            repeats += 2
        grid = DET_SIZE // strides[layer]
        for y in range(grid):
            for x in range(grid):
                cx = (x + 0.5) / grid
                cy = (y + 0.5) / grid
                for _ in range(repeats):
                    anchors.append((cx, cy))
        layer = last
    return np.asarray(anchors, dtype=np.float32)


ANCHORS = _ssd_anchors()


@dataclass
class TensorSpec:
    index: int
    dtype: type
    scale: float
    zero_point: int

    @classmethod
    def from_details(cls, details: dict) -> TensorSpec:
        scale, zp = details.get("quantization", (0.0, 0))
        return cls(
            index=int(details["index"]),
            dtype=details["dtype"],
            scale=float(scale or 0.0),
            zero_point=int(zp or 0),
        )

    def encode(self, x: np.ndarray) -> np.ndarray:
        if self.dtype == np.float32:
            return x if x.dtype == np.float32 else x.astype(np.float32)
        x = np.asarray(x, dtype=np.float32)
        if self.scale == 0.0:
            return x.astype(self.dtype)
        q = np.round(x / self.scale + self.zero_point)
        info = np.iinfo(self.dtype)
        return np.clip(q, info.min, info.max).astype(self.dtype)

    def decode(self, y: np.ndarray) -> np.ndarray:
        if self.dtype == np.float32:
            return y
        return (y.astype(np.float32) - self.zero_point) * self.scale

    def float_range(self, default: tuple[float, float]) -> tuple[float, float]:
        if self.dtype == np.float32 or self.scale == 0.0:
            return default
        info = np.iinfo(self.dtype)
        return (
            (info.min - self.zero_point) * self.scale,
            (info.max - self.zero_point) * self.scale,
        )


def _letterbox_rgb(
    rgb: np.ndarray, size: int, lo: float = -1.0, hi: float = 1.0
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Resize with aspect ratio, pad to square. Pad with mid-value of [lo, hi]."""
    h, w = rgb.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    resized = resized / 255.0 * (hi - lo) + lo
    mid = 0.5 * (lo + hi)
    canvas = np.full((size, size, 3), mid, dtype=np.float32)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    pad = (
        left / size,
        top / size,
        (size - left - nw) / size,
        (size - top - nh) / size,
    )
    return canvas, pad


def _unletterbox(xy: np.ndarray, pad: tuple[float, float, float, float]) -> np.ndarray:
    left, top, right, bottom = pad
    out = xy.copy()
    out[..., 0] = (out[..., 0] - left) / max(1e-6, 1.0 - left - right)
    out[..., 1] = (out[..., 1] - top) / max(1e-6, 1.0 - top - bottom)
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -RAW_SCORE_LIMIT, RAW_SCORE_LIMIT)
    return 1.0 / (1.0 + np.exp(-x))


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    denom = area + areas - inter
    return np.where(denom > 0, inter / denom, 0.0)


def _weighted_nms(
    boxes: np.ndarray, scores: np.ndarray, kps: np.ndarray
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    order = scores.argsort()[::-1]
    kept: list[tuple[np.ndarray, float, np.ndarray]] = []
    while order.size:
        i = int(order[0])
        rest = order[1:]
        if rest.size:
            ious = _iou(boxes[i], boxes[rest])
            similar = np.where(ious > NMS_THRESH)[0]
            group = np.concatenate(([i], rest[similar]))
            remain = rest[np.where(ious <= NMS_THRESH)[0]]
        else:
            group = np.array([i])
            remain = rest
        w = scores[group]
        wsum = float(w.sum())
        box = (boxes[group] * w[:, None]).sum(axis=0) / wsum
        kp = (kps[group] * w[:, None, None]).sum(axis=0) / wsum
        kept.append((box, float(scores[i]), kp))
        order = remain
    return kept


@dataclass
class FaceDet:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    score: float
    keypoints: np.ndarray  # (6, 2) normalized


class FacePipeline:
    def __init__(self, det_path: Path = DET_MODEL, lm_path: Path = LM_MODEL) -> None:
        self.det_path = Path(det_path)
        self.lm_path = Path(lm_path)
        self.det = make_interpreter(self.det_path)
        self.lm = make_interpreter(self.lm_path)
        self.det_in = TensorSpec.from_details(self.det.get_input_details()[0])
        self.lm_in = TensorSpec.from_details(self.lm.get_input_details()[0])
        self._bind_det_outputs()
        self._bind_lm_outputs()

    def _bind_det_outputs(self) -> None:
        boxes = scores = None
        for d in self.det.get_output_details():
            shape = tuple(int(x) for x in d["shape"])
            spec = TensorSpec.from_details(d)
            if shape[-1] == 16:
                boxes = spec
            elif shape[-1] == 1:
                scores = spec
        if boxes is None or scores is None:
            raise RuntimeError(f"unexpected det outputs: {self.det.get_output_details()}")
        self.det_boxes = boxes
        self.det_scores = scores

    def _bind_lm_outputs(self) -> None:
        data = flag = None
        for d in self.lm.get_output_details():
            n = int(np.prod(d["shape"]))
            spec = TensorSpec.from_details(d)
            if n >= NUM_LANDMARKS * 3:
                data = spec
            elif n <= 4:
                flag = spec
        if data is None or flag is None:
            raise RuntimeError(f"unexpected lm outputs: {self.lm.get_output_details()}")
        self.lm_data = data
        self.lm_flag = flag

    def preprocess_det(self, bgr: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lo, hi = self.det_in.float_range((-1.0, 1.0))
        tensor, pad = _letterbox_rgb(rgb, DET_SIZE, lo, hi)
        return tensor[np.newaxis], pad

    def infer_det(self, inp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.det.set_tensor(self.det_in.index, self.det_in.encode(inp))
        self.det.invoke()
        return (
            self.det_boxes.decode(self.det.get_tensor(self.det_boxes.index)),
            self.det_scores.decode(self.det.get_tensor(self.det_scores.index)),
        )

    def post_det(
        self, raw_boxes: np.ndarray, raw_scores: np.ndarray, pad: tuple[float, float, float, float]
    ) -> list[FaceDet]:
        boxes = raw_boxes.reshape(-1, 8, 2).astype(np.float32) / float(DET_SIZE)
        boxes[:, 0] += ANCHORS
        boxes[:, 2:] += ANCHORS[:, None, :]
        center = boxes[:, 0]
        half = boxes[:, 1] / 2.0
        xyxy = np.concatenate([center - half, center + half], axis=1)
        kps = boxes[:, 2:]  # (N, 6, 2)
        scores = _sigmoid(raw_scores.reshape(-1))
        mask = scores > MIN_SCORE
        if not np.any(mask):
            return []
        xyxy = _unletterbox(xyxy.reshape(-1, 2), pad).reshape(-1, 4)[mask]
        kps = _unletterbox(kps, pad)[mask]
        scores = scores[mask]
        valid = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
        xyxy, scores, kps = xyxy[valid], scores[valid], kps[valid]
        if xyxy.size == 0:
            return []
        merged = _weighted_nms(xyxy, scores, kps)
        out: list[FaceDet] = []
        for box, score, kp in merged:
            out.append(
                FaceDet(
                    xmin=float(box[0]),
                    ymin=float(box[1]),
                    xmax=float(box[2]),
                    ymax=float(box[3]),
                    score=score,
                    keypoints=kp,
                )
            )
        return out

    def detect(self, bgr: np.ndarray) -> list[FaceDet]:
        inp, pad = self.preprocess_det(bgr)
        raw_boxes, raw_scores = self.infer_det(inp)
        return self.post_det(raw_boxes, raw_scores, pad)

    def face_roi(self, det: FaceDet, image_wh: tuple[int, int]) -> tuple[float, float, float, float]:
        """Return (cx_px, cy_px, side_px, rotation_rad) for the landmark crop."""
        w_img, h_img = image_wh
        abs_w = (det.xmax - det.xmin) * w_img
        abs_h = (det.ymax - det.ymin) * h_img
        side = max(abs_w, abs_h) * ROI_SCALE
        cx = (det.xmin + det.xmax) * 0.5 * w_img
        cy = (det.ymin + det.ymax) * 0.5 * h_img
        # kp0 = right eye, kp1 = left eye (MediaPipe short-range)
        x0, y0 = det.keypoints[0]
        x1, y1 = det.keypoints[1]
        x0, y0 = x0 * w_img, y0 * h_img
        x1, y1 = x1 * w_img, y1 * h_img
        angle = -math.atan2(y0 - y1, x1 - x0)
        two_pi = 2.0 * math.pi
        rot = angle - two_pi * math.floor((angle + math.pi) / two_pi)
        return cx, cy, side, rot

    def crop_face(
        self, bgr: np.ndarray, roi: tuple[float, float, float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        cx, cy, side, rot = roi
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        hw = side / 2.0

        def corner(dx: float, dy: float) -> tuple[float, float]:
            return cx + dx * cos_r - dy * sin_r, cy + dx * sin_r + dy * cos_r

        src = np.array(
            [corner(-hw, -hw), corner(hw, -hw), corner(hw, hw)],
            dtype=np.float32,
        )
        dst = np.array(
            [[0.0, 0.0], [LM_SIZE, 0.0], [LM_SIZE, LM_SIZE]],
            dtype=np.float32,
        )
        m = cv2.getAffineTransform(src, dst)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        crop = cv2.warpAffine(rgb, m, (LM_SIZE, LM_SIZE), flags=cv2.INTER_LINEAR)
        lo, hi = self.lm_in.float_range((0.0, 1.0))
        tensor = crop.astype(np.float32) / 255.0 * (hi - lo) + lo
        return tensor[np.newaxis], m

    def infer_lm(self, inp: np.ndarray) -> tuple[np.ndarray, float]:
        self.lm.set_tensor(self.lm_in.index, self.lm_in.encode(inp))
        self.lm.invoke()
        raw = self.lm_data.decode(self.lm.get_tensor(self.lm_data.index)).reshape(-1)
        flag = float(_sigmoid(self.lm_flag.decode(self.lm.get_tensor(self.lm_flag.index)).reshape(-1))[-1])
        return raw[: NUM_LANDMARKS * 3].reshape(NUM_LANDMARKS, 3), flag

    def project_landmarks(self, lms: np.ndarray, affine: np.ndarray) -> np.ndarray:
        inv = cv2.invertAffineTransform(affine)
        xy = lms[:, :2]
        ones = np.ones((xy.shape[0], 1), dtype=np.float32)
        pix = np.concatenate([xy, ones], axis=1) @ inv.T
        out = lms.copy()
        out[:, 0] = pix[:, 0]
        out[:, 1] = pix[:, 1]
        return out

    def landmarks(self, bgr: np.ndarray, det: FaceDet) -> np.ndarray | None:
        h, w = bgr.shape[:2]
        roi = self.face_roi(det, (w, h))
        inp, affine = self.crop_face(bgr, roi)
        lms, flag = self.infer_lm(inp)
        if flag <= FACE_FLAG_THRESH:
            return None
        return self.project_landmarks(lms, affine)
