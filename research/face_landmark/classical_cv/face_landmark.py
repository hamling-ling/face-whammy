#!/usr/bin/env python3
from pathlib import Path
import cv2

IMAGE, MODEL, OUT = Path("natasha.jpg"), Path("lbfmodel.yaml"), Path("natasha_landmarks.jpg")

img = cv2.imread(str(IMAGE))
if img is None:
    raise FileNotFoundError(IMAGE.resolve())

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
).detectMultiScale(gray, 1.1, 5)
print("faces:", len(faces))
if len(faces) == 0:
    raise SystemExit("顔なし")

if not hasattr(cv2, "face"):
    raise SystemExit("cv2.face なし。opencv-contrib-python-headless を入れてください")

facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(str(MODEL))
ok, landmarks = facemark.fit(img, faces)
if not ok:
    raise SystemExit("landmark 失敗")

for lm in landmarks:
    for x, y in lm[0]:
        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)

cv2.imwrite(str(OUT), img)
print("saved", OUT.resolve(), "points", len(landmarks[0][0]))

