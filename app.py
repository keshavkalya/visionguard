import cv2
import numpy as np
import json
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

THRESHOLD = 60  # LBPH distance cutoff, same as visionguard.py

# Load the face detector (finds WHERE faces are)
detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx", "", (320, 320), 0.8
)

# Load the trained recognizer (tells WHO the face is)
if not os.path.exists("trainer.yml"):
    print("Run train.py first")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Load the ID -> name mapping, e.g. {"0": "alice"}
with open("names.json") as f:
    names = json.load(f)


def read_uploaded_image(file):
    # Turn the uploaded file into an OpenCV image (BGR), or None if it's junk
    data = file.read()
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def detect_best_face(img):
    # Returns (x, y, w, h) of the most confident face, or None if no face found
    h_frame, w_frame = img.shape[:2]
    detector.setInputSize((w_frame, h_frame))
    _, detections = detector.detect(img)

    if detections is None or len(detections) == 0:
        return None

    best = max(detections, key=lambda d: d[-1])
    x, y, w, h = map(int, best[:4])
    x, y = max(0, x), max(0, y)
    return x, y, w, h


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "known_people": list(names.values())})


@app.route("/verify", methods=["POST"])
def verify():
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file field"}), 400

    img = read_uploaded_image(request.files["image"])
    if img is None:
        return jsonify({"error": "could not read image"}), 400

    box = detect_best_face(img)
    if box is None:
        return jsonify({"authorized": False, "reason": "no_face_detected"})

    x, y, w, h = box
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_img = gray[y:y+h, x:x+w]
    if face_img.size == 0:
        return jsonify({"authorized": False, "reason": "no_face_detected"})
    face_img = cv2.resize(face_img, (200, 200))

    # Ask the recognizer: who is this?
    label_id, distance = recognizer.predict(face_img)

    # LBPH gives a DISTANCE: lower = better match. ~0-60 is a good match.
    if distance < THRESHOLD:
        name = names[str(label_id)]
        return jsonify({"authorized": True, "name": name, "distance": float(distance)})
    else:
        return jsonify({"authorized": False, "reason": "unknown_face"})


@app.route("/enroll", methods=["POST"])
def enroll():
    name = request.form.get("name", "").strip().lower()
    if not name:
        return jsonify({"error": "missing 'name' field"}), 400

    files = request.files.getlist("image")
    if not files:
        return jsonify({"error": "missing 'image' file field"}), 400

    folder = os.path.join("dataset", name)
    os.makedirs(folder, exist_ok=True)

    # Start numbering after whatever photos this person already has
    count = len([f for f in os.listdir(folder) if f.endswith(".jpg")])

    saved = 0
    for file in files:
        img = read_uploaded_image(file)
        if img is None:
            continue  # skip unreadable file

        box = detect_best_face(img)
        if box is None:
            continue  # skip images with no face in them

        x, y, w, h = box
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_img = gray[y:y+h, x:x+w]
        if face_img.size == 0:
            continue
        face_img = cv2.resize(face_img, (200, 200))

        count += 1
        cv2.imwrite(os.path.join(folder, f"{count}.jpg"), face_img)
        saved += 1

    if saved == 0:
        return jsonify({"error": "no faces found in uploaded image(s)"}), 400

    retrain()

    return jsonify({"enrolled": name, "images_saved": saved})


def retrain():
    # Same training logic as train.py, just re-run here so the API stays up to date
    global names

    dataset_path = "dataset"
    faces = []
    labels = []
    new_names = {}

    for label_id, person_name in enumerate(sorted(os.listdir(dataset_path))):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue  # skip stray files like .DS_Store

        new_names[label_id] = person_name

        for filename in os.listdir(person_folder):
            img_path = os.path.join(person_folder, filename)
            face = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if face is None:
                continue
            face = cv2.resize(face, (200, 200))
            faces.append(face)
            labels.append(label_id)

    recognizer.train(faces, np.array(labels))
    recognizer.save("trainer.yml")

    with open("names.json", "w") as f:
        json.dump(new_names, f)

    # Keep the in-memory copy in sync so /health and /verify see the update right away
    names = {str(k): v for k, v in new_names.items()}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
