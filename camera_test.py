
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("无法打开摄像头，请检查摄像头权限")

while True:
    ret, frame = cap.read()

    if not ret:
        print("读取摄像头画面失败")
        break

    frame = cv2.flip(frame, 1)

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
