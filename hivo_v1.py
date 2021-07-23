import alvonCV
import cv2
import time

cap = cv2.VideoCapture(0)
pTime = 0

# here you use the alvonCV package
faceDetectorObj = alvonCV.FaceDetector()

while True:
    success, img = cap.read()
    img, bboxs = faceDetectorObj.findFaces(img, draw=True)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
