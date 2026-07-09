import cv2
import os

# Ask whose face we're capturing
name = input("Enter your name: ").strip().lower()

# Create a folder like dataset/keshav to store the photos
folder = os.path.join("dataset", name)
os.makedirs(folder, exist_ok=True)

# Modern deep-learning face detector (YuNet)
# 0.8 = only report faces it is 80%+ confident about
detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx", "", (320, 320), 0.8
)

camera = cv2.VideoCapture(0)
count = 0  # how many face photos saved so far

while count < 20:
    success, frame = camera.read()
    if not success:
        break

    # Tell the detector the size of this frame, then detect
    h_frame, w_frame = frame.shape[:2]
    detector.setInputSize((w_frame, h_frame))
    _, detections = detector.detect(frame)

    if detections is not None and len(detections) > 0:
        # Pick the detection with the highest confidence score
        best = max(detections, key=lambda d: d[-1])
        x, y, w, h = map(int, best[:4])

        # Safety: keep the box inside the frame edges
        x, y = max(0, x), max(0, y)

        count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(folder, f"{count}.jpg"), face_img)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Captured: {count}/20 - move your head slowly", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Enrolling - look at camera", frame)
    if cv2.waitKey(700) & 0xFF == ord("q"):  # ~0.7s gap between captures
        break

camera.release()
cv2.destroyAllWindows()
print(f"Saved {count} images to {folder}")