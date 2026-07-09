import cv2
import json

# Load the face detector (finds WHERE faces are)
detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx", "", (320, 320), 0.8
)

# Load the trained recognizer (tells WHO the face is)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Load the ID -> name mapping, e.g. {"0": "alice"}
with open("names.json") as f:
    names = json.load(f)

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    if not success:
        break

    h_frame, w_frame = frame.shape[:2]
    detector.setInputSize((w_frame, h_frame))
    _, detections = detector.detect(frame)

    if detections is not None:
        for det in detections:
            x, y, w, h = map(int, det[:4])
            x, y = max(0, x), max(0, y)

            # Crop the face and prepare it exactly like training data
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_img = gray[y:y+h, x:x+w]
            if face_img.size == 0:
                continue
            face_img = cv2.resize(face_img, (200, 200))

            # Ask the recognizer: who is this?
            label_id, distance = recognizer.predict(face_img)

            # LBPH gives a DISTANCE: lower = more similar. ~0-60 is a good match.
            if distance < 60:
                name = names[str(label_id)]
                color = (0, 255, 0)  # green = recognized
                text = f"{name} ({distance:.0f})"
            else:
                name = "Unknown"
                color = (0, 0, 255)  # red = stranger
                text = f"Unknown ({distance:.0f})"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("VisionGuard - Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()