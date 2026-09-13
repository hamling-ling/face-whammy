#!/usr/bin/env python3
import argparse
from pathlib import Path
import statistics
import time

import cv2

from pipeline import QUANT_MODELS, FacePipeline

HERE = Path(__file__).resolve().parent
IMAGE = HERE / "natasha.jpg"
N_WARMUP = 3
N_RUN = 20


def ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def summarize(name: str, xs: list[float]) -> None:
    print(
        f"{name:12s}  mean={statistics.mean(xs):7.1f} ms  "
        f"median={statistics.median(xs):7.1f} ms  "
        f"min={min(xs):7.1f}  max={max(xs):7.1f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=sorted(QUANT_MODELS), default="fp32")
    args = ap.parse_args()

    img = cv2.imread(str(IMAGE))
    if img is None:
        raise SystemExit(f"cannot read {IMAGE}")
    print(f"image: {img.shape[1]}x{img.shape[0]}  runs={N_RUN}  quant={args.quant}")

    det_path, lm_path = QUANT_MODELS[args.quant]
    t0 = time.perf_counter()
    pipe = FacePipeline(det_path, lm_path)
    print(f"loadModel: {ms(t0):.1f} ms")
    print("det:", pipe.det_path.name, pipe.det_in.dtype.__name__)
    print("lm: ", pipe.lm_path.name, pipe.lm_in.dtype.__name__)

    for _ in range(N_WARMUP):
        faces = pipe.detect(img)
        if faces:
            pipe.landmarks(img, faces[0])

    t_pre, t_det, t_post, t_crop, t_lm, t_all = [], [], [], [], [], []
    last_faces = None
    last_pts = 0
    for _ in range(N_RUN):
        t_all0 = time.perf_counter()

        t0 = time.perf_counter()
        inp, pad = pipe.preprocess_det(img)
        t_pre.append(ms(t0))

        t0 = time.perf_counter()
        raw_boxes, raw_scores = pipe.infer_det(inp)
        t_det.append(ms(t0))

        t0 = time.perf_counter()
        faces = pipe.post_det(raw_boxes, raw_scores, pad)
        t_post.append(ms(t0))
        last_faces = faces

        if not faces:
            t_crop.append(0.0)
            t_lm.append(0.0)
            t_all.append(ms(t_all0))
            continue

        det = faces[0]
        t0 = time.perf_counter()
        roi = pipe.face_roi(det, (img.shape[1], img.shape[0]))
        crop, affine = pipe.crop_face(img, roi)
        t_crop.append(ms(t0))

        t0 = time.perf_counter()
        lms, flag = pipe.infer_lm(crop)
        if flag > 0.5:
            lms = pipe.project_landmarks(lms, affine)
            last_pts = len(lms)
        t_lm.append(ms(t0))

        t_all.append(ms(t_all0))

    print("faces:", len(last_faces) if last_faces is not None else 0, "points:", last_pts)
    summarize("preprocess", t_pre)
    summarize("BlazeFace", t_det)
    summarize("det post", t_post)
    summarize("crop", t_crop)
    summarize("FaceMesh", t_lm)
    summarize("total", t_all)
    mean = statistics.mean(t_all)
    print(f"approx FPS (this image): {1000.0 / mean:.1f}")


if __name__ == "__main__":
    main()
