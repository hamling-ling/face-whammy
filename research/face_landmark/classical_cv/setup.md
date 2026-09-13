```
wget -O haarcascade_frontalface_default.xml \
  https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml

wget -O lbfmodel.yaml \
  https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml

# contrib 必須（cv2.face）
pip install opencv-contrib-python-headless
```

run
```
python3 face_landmark.py
```

```
(.venv-face) arduino@uno-q-1:~/work/face_landmark$ python3 run_landmark.py 
image: 670x377  runs=20
loading data from : /home/arduino/work/face_landmark/lbfmodel.yaml
loadModel: 3156.2 ms
faces: 1
cvtColor      mean=    0.3 ms  median=    0.3 ms  min=    0.3  max=    0.7
Haar          mean=   44.0 ms  median=   41.2 ms  min=   41.0  max=   72.7
LBF fit       mean=   37.9 ms  median=   37.9 ms  min=   37.3  max=   39.2
total         mean=   82.2 ms  median=   79.4 ms  min=   78.8  max=  112.2
approx FPS (this image): 12.2
```
