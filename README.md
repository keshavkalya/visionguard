# VisionGuard

VisionGuard is a small face recognition access control system I built while
learning computer vision with OpenCV. It detects a face with a deep learning
model (YuNet), recognizes who it is with an LBPH recognizer trained on my own
face photos, and checks for a blink before granting access, so a printed
photo can't fool it. There's also a Flask API on top so you can enroll people
and verify faces over HTTP instead of just from the webcam scripts.

## Features

- Face detection with YuNet (a real deep learning model, not just Haar cascades)
- Face recognition with LBPH (trained on photos you capture yourself)
- Liveness check via blink detection (Haar eye cascade) so a photo of a photo doesn't pass
- Enroll new people from the webcam (`enroll.py`) or via the API (`/enroll`)
- Combined "ACCESS GRANTED / ACCESS DENIED" webcam demo (`visionguard.py`)
- REST API (`app.py`) to verify and enroll faces without opening a webcam window

## Tech stack

- Python
- OpenCV (`opencv-contrib-python`) — YuNet face detector + LBPH face recognizer
- Flask — REST API
- NumPy

## Setup

Clone the repo, then set up a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the YuNet face detection model (it's not committed to the repo):

```bash
curl -L -o face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

Enroll at least one person (this opens your webcam and captures 20 face photos):

```bash
python enroll.py
```

Train the recognizer on whoever you've enrolled so far:

```bash
python train.py
```

Then either run the webcam demo:

```bash
python visionguard.py
```

or start the API:

```bash
python app.py
```

## API

The API loads the detector, recognizer, and known names once at startup, so
make sure you've run `train.py` at least once before starting it. It runs on
port 5001 (5000 clashes with AirPlay Receiver on macOS).

### GET /health

Check the server is up and see who it currently knows.

```bash
curl http://localhost:5001/health
```

```json
{"status": "ok", "known_people": ["alice", "bob"]}
```

### POST /verify

Upload an image and it'll tell you if it recognizes the face in it.

```bash
curl -X POST http://localhost:5001/verify -F "image=@test.jpg"
```

Recognized:

```json
{"authorized": true, "name": "alice", "distance": 18.46}
```

Not recognized:

```json
{"authorized": false, "reason": "unknown_face"}
```

No face found in the image:

```json
{"authorized": false, "reason": "no_face_detected"}
```

### POST /enroll

Upload a name and one or more photos of that person. It'll crop the face out
of each photo, save it to `dataset/<name>/`, and retrain automatically.

```bash
curl -X POST http://localhost:5001/enroll -F "name=alice" -F "image=@photo1.jpg" -F "image=@photo2.jpg"
```

```json
{"enrolled": "alice", "images_saved": 2}
```

### Quick demo script

`demo.sh` runs all three endpoints in one go using a photo you give it —
health check, enroll the photo as `demo_user`, then verify the same photo:

```bash
sh demo.sh path/to/your/photo.jpg
```

(To undo the demo enrollment afterwards: delete `dataset/demo_user/` and
re-run `train.py`.)

### Testing /verify yourself

Take a clear photo of your face with your phone or webcam app and save it
somewhere, e.g. `test.jpg`, then run:

```bash
curl -X POST http://localhost:5001/verify -F "image=@test.jpg"
```

You can also just reuse one of your own enrollment photos from
`dataset/<your_name>/1.jpg` to sanity check that it recognizes you.

## How it works

1. **Detection** — YuNet looks at the frame and returns bounding boxes for
   any faces it finds, plus a confidence score. We keep the most confident one.
2. **Recognition** — the face is cropped, converted to grayscale, resized to
   200x200 (same size used for training), and fed into the LBPH recognizer.
   LBPH compares local pixel patterns against what it learned from your
   enrollment photos and returns a distance — lower distance means a better
   match. Anything under 60 is treated as a match.
3. **Liveness** — a Haar eye cascade checks the top half of the face box for
   eyes. If eyes disappear for a few frames and then come back, that's
   counted as a blink. `visionguard.py` only grants access once the face is
   both recognized AND a blink has happened.

## Limitations

- LBPH is sensitive to lighting — recognition accuracy drops a lot in dim or
  uneven light compared to how you enrolled.
- Blink detection needs decent lighting too, since it relies on the Haar eye
  cascade actually finding your eyes.
- It's trained on a small number of photos per person (20), so it won't
  generalize as well as a proper deep learning face recognition model would.
- No anti-spoofing beyond the blink check — it's not resistant to a video
  replay of someone blinking.
- The distance threshold (60) was picked by eye, not tuned scientifically.

## Demo

<!-- TODO: add demo gif here -->
