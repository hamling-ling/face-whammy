#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2

from pipeline import QUANT_MODELS, FacePipeline

HERE = Path(__file__).resolve().parent
IMAGE = HERE / "natasha.jpg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=sorted(QUANT_MODELS), default="fp32")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or HERE / f"natasha_landmarks_{args.quant}.jpg"
    if args.quant == "fp32" and args.out is None:
        out = HERE / "natasha_landmarks.jpg"

    img = cv2.imread(str(IMAGE))
    if img is None:
        raise FileNotFoundError(IMAGE.resolve())

    det_path, lm_path = QUANT_MODELS[args.quant]
    pipe = FacePipeline(det_path, lm_path)
    print("quant:", args.quant)
    print("det:", pipe.det_path.name, pipe.det_in.dtype.__name__)
    print("lm: ", pipe.lm_path.name, pipe.lm_in.dtype.__name__)

    faces = pipe.detect(img)
    print("faces:", len(faces))
    if not faces:
        raise SystemExit("顔なし")

    vis = img.copy()
    n_pts = 0
    for det in faces:
        x1, y1 = int(det.xmin * img.shape[1]), int(det.ymin * img.shape[0])
        x2, y2 = int(det.xmax * img.shape[1]), int(det.ymax * img.shape[0])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)
        lms = pipe.landmarks(img, det)
        if lms is None:
            continue
        n_pts = len(lms)
        for x, y, _z in lms:
            cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 0), -1)

    cv2.imwrite(str(out), vis)
    print("saved", out.resolve(), "points", n_pts)


if __name__ == "__main__":
    main()
