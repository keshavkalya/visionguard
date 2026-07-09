import cv2

# Modern deep-learning face detector (YuNet)
detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx", "", (320, 320), 0.8
)

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    if not success:
        break

    # Tell the detector the size of this frame, then detect
    h_frame, w_frame = frame.shape[:2]
    detector.setInputSize((w_frame, h_frame))
    _, detections = detector.detect(frame)

    if detections is not None:
        for det in detections:
            x, y, w, h = map(int, det[:4])
            confidence = det[-1]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {confidence:.0%}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("VisionGuard", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()