Arduino UNO Q (Debian / aarch64) 上で MediaPipe の公式 TFLite を LiteRT で動かす。

モデル
```
mkdir -p models
wget -O models/face_detection_short_range.tflite \
  https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite
wget -O models/face_landmark.tflite \
  https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite
```

サンプル画像
```
curl -L -o natasha.jpg 'https://images.bauerhosting.com/legacy/media/619a/954a/3ebe/4782/749c/bee1/Natasha%20Liu%20Bordizzo.jpg'
```

- `face_detection_short_range.tflite` — BlazeFace 128×128、近距離向け（〜2 m）
- `face_landmark.tflite` — Face Mesh 192×192、468 点

venv（古典 CV と同じ `~/work/.venv-face` を流用可）
```
python3 -m venv ~/work/.venv-face
source ~/work/.venv-face/bin/activate
pip install ai-edge-litert opencv-contrib-python-headless
```

board へ
```
scp -r research/face_landmark/tflite arduino@uno-q-1.local:~/work/face_landmark_tflite
```

run
```
source ~/work/.venv-face/bin/activate
cd ~/work/face_landmark_tflite
python3 face_landmark.py
python3 run_landmark.py
```

UNO Q (`uno-q-1`, QRB2210 / 4×A53) 上の結果（`natasha.jpg` 670×377, 20 runs）
```
(.venv-face) arduino@uno-q-1:~/work/face_landmark_tflite$ python3 run_landmark.py
image: 670x377  runs=20
loadModel: 75.2 ms
faces: 1 points: 468
preprocess    mean=    1.2 ms  median=    1.2 ms  min=    1.1  max=    1.4
BlazeFace     mean=    6.6 ms  median=    6.5 ms  min=    6.3  max=    7.3
det post      mean=    2.0 ms  median=    2.1 ms  min=    2.0  max=    2.1
crop          mean=    1.7 ms  median=    1.7 ms  min=    1.4  max=    2.4
FaceMesh      mean=   10.0 ms  median=    9.8 ms  min=    9.5  max=   11.4
total         mean=   21.6 ms  median=   21.5 ms  min=   20.6  max=   23.9
approx FPS (this image): 46.3
```

古典 CV（Haar + LBF, 同画像・同ボード）は total ~82 ms / ~12 FPS。TFLite は検出 6.6 ms + ランドマーク 10 ms で、点は 68 → 468。

INT8（PINTO integer_quant。I/O は float32 ラッパ、中の conv は int8）
```
python3 face_landmark.py --quant int8
python3 run_landmark.py --quant int8
```

同ボード連続計測（20 runs）:

| | FP32 公式 | INT8 |
|---|---|---|
| det モデル | 229 KB | 197 KB |
| lm モデル | 1.2 MB（内部は float16 重み） | 778 KB |
| BlazeFace | 9.1 ms | 6.8 ms |
| FaceMesh | 12.9 ms | 10.0 ms |
| total | 28.6 ms (~35 FPS) | 24.0 ms (~42 FPS) |

可視化は `natasha_landmarks_int8.jpg`。468 点は FP32 と同位置に乗る。このサイズでは INT8 の得はモデル縮小が主で、A53 + XNNPACK の推論時間は同程度〜やや速い。
