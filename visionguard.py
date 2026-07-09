import cv2
import json

# --- Load all three brains ---
detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx", "", (320, 320), 0.8
)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
with open("names.json") as f:
    names = json.load(f)

THRESHOLD = 60  # LBPH distance cutoff (tune this to your tested value)

camera = cv2.VideoCapture(0)

eyes_were_visible = False
closed_frames = 0
blinks = 0

while True:
    success, frame = camera.read()
    if not success:
        break

    h_frame, w_frame = frame.shape[:2]
    detector.setInputSize((w_frame, h_frame))
    _, detections = detector.detect(frame)

    if detections is not None and len(detections) > 0:
        best = max(detections, key=lambda d: d[-1])
        x, y, w, h = map(int, best[:4])
        x, y = max(0, x), max(0, y)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- 1. WHO is it? ---
        face_img = gray[y:y+h, x:x+w]
        if face_img.size == 0:
            continue
        face_img = cv2.resize(face_img, (200, 200))
        label_id, distance = recognizer.predict(face_img)
        recognized = distance < THRESHOLD
        name = names[str(label_id)] if recognized else "Unknown"

        # --- 2. Is it ALIVE? (blink check) ---
        face_top = gray[y:y + h // 2, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_top, 1.1, 7, minSize=(30, 30))
        eyes_visible = len(eyes) >= 1

        if eyes_visible:
            if not eyes_were_visible and 1 <= closed_frames <= 5:
                blinks += 1
            closed_frames = 0
        else:
            closed_frames += 1
        eyes_were_visible = eyes_visible

        is_live = blinks > 0

        # --- 3. Final decision: must be BOTH recognized AND live ---
        if recognized and is_live:
            color = (0, 255, 0)          # green
            status = f"ACCESS GRANTED: {name}"
        elif recognized and not is_live:
            color = (0, 165, 255)        # orange
            status = f"{name} - blink to confirm"
        else:
            color = (0, 0, 255)          # red
            status = "ACCESS DENIED: Unknown"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, status, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("VisionGuard", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()