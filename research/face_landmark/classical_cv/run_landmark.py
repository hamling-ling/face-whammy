#!/usr/bin/env python3
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
IMAGE = HERE / "natasha.jpg"
CASCADE = HERE / "haarcascade_frontalface_default.xml"
MODEL = HERE / "lbfmodel.yaml"
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
    img = cv2.imread(str(IMAGE))
    if img is None:
        raise SystemExit(f"cannot read {IMAGE}")
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"image: {img.shape[1]}x{img.shape[0]}  runs={N_RUN}")

    det = cv2.CascadeClassifier(str(CASCADE))
    facemark = cv2.face.createFacemarkLBF()
    t0 = time.perf_counter()
    facemark.loadModel(str(MODEL))
    print(f"loadModel: {ms(t0):.1f} ms")

    for _ in range(N_WARMUP):
        faces = det.detectMultiScale(gray0, 1.1, 5, minSize=(60, 60))
        facemark.fit(img, faces)

    t_cvt, t_haar, t_lbf, t_all = [], [], [], []
    last_faces = None
    for _ in range(N_RUN):
        t_all0 = time.perf_counter()

        t0 = time.perf_counter()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t_cvt.append(ms(t0))

        t0 = time.perf_counter()
        faces = det.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        t_haar.append(ms(t0))
        last_faces = faces

        t0 = time.perf_counter()
        ok, lms = facemark.fit(img, faces)
        t_lbf.append(ms(t0))

        t_all.append(ms(t_all0))

    print("faces:", len(last_faces) if last_faces is not None else 0)
    summarize("cvtColor", t_cvt)
    summarize("Haar", t_haar)
    summarize("LBF fit", t_lbf)
    summarize("total", t_all)
    mean = statistics.mean(t_all)
    print(f"approx FPS (this image): {1000.0 / mean:.1f}")


if __name__ == "__main__":
    main()
