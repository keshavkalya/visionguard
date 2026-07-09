import cv2

# Face detector (YuNet) + eye detector (Haar, ships with OpenCV)
detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx", "", (320, 320), 0.8
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

camera = cv2.VideoCapture(0)

eyes_were_visible = False  # were eyes visible in the previous frame?
closed_frames = 0          # how many frames in a row with no eyes
blinks = 0                 # total blinks counted

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

        # Look for eyes ONLY in the top half of the face box
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_top = gray[y:y + h // 2, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_top, 1.1, 7, minSize=(30, 30))

        eyes_visible = len(eyes) >= 1

        if eyes_visible:
            # Eyes came back after 1-5 frames of absence = that was a blink
            if not eyes_were_visible and 1 <= closed_frames <= 5:
                blinks += 1
            closed_frames = 0
        else:
            closed_frames += 1

        eyes_were_visible = eyes_visible

        color = (0, 255, 0) if blinks > 0 else (0, 165, 255)
        status = f"LIVE (blinks: {blinks})" if blinks > 0 else "Blink to prove you're real"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, status, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("VisionGuard - Liveness", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()