import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox)

                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Top Left  x,y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
        # Top Right  x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # Bottom Left  x,y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img


class FaceLocation:
    def __init__(self):
        pass

    def drawSection(self, img, bboxs):
        img_height, img_width, img_channel = img.shape

        ih, iw, ic = img.shape

        max_area = 0
        max_area_idx = 0
        max_area_mdl_width = 0
        max_area_mdl_height = 0
        for index, b in enumerate(bboxs):
            bounding_box = b[1]
            c_area = bounding_box[2] * bounding_box[3]
            if c_area > max_area:
                max_area = c_area
                max_area_idx = index
                max_area_mdl_width = int(bounding_box[0] + bounding_box[2] / 2)
                max_area_mdl_height = int(bounding_box[1] + bounding_box[3] / 2)

        for index, b in enumerate(bboxs):
            bounding_box = b[1]
            c_area = bounding_box[2] * bounding_box[3]
            if c_area > max_area:
                max_area = c_area
                max_area_idx = index
                max_area_mdl_width = int(bounding_box[0] + bounding_box[2] / 2)
                max_area_mdl_height = int(bounding_box[1] + bounding_box[3] / 2)

        # cv2.circle(img, (max_area_mdl_width, max_area_mdl_height), 3, (0, 0, 100), 2)

        section = ''
        # middle line
        cv2.line(img, (img_width // 2, 0), (img_width // 2, img_height), (255, 0, 100), 2)

        per_3_line = int(0.03 * img_width)
        # left line
        portionILeft = img_width // 2 - per_3_line
        cv2.line(img, (portionILeft, 0), (portionILeft, img_height), (0, 100, 50), 2)
        if max_area_mdl_width < portionILeft:
            section = 'C'
        per_15_line = int(0.15 * img_width)
        # left line
        portionIILeft = img_width // 2 - per_15_line
        cv2.line(img, (portionIILeft, 0), (portionIILeft, img_height), (50, 0, 100), 2)
        if max_area_mdl_width < portionIILeft:
            section = 'B'
        per_40_line = int(0.40 * img_width)
        # left line
        portionIIILeft = img_width // 2 - per_40_line
        cv2.line(img, (portionIIILeft, 0), (portionIIILeft, img_height), (100, 0, 0), 2)
        if max_area_mdl_width < portionIIILeft:
            section = 'A'


        # right line
        portionIRight = img_width // 2 + per_3_line
        cv2.line(img, (portionIRight, 0), (portionIRight, img_height), (0, 100, 50), 2)
        if max_area_mdl_width > portionIRight:
            section = 'E'
        # right line
        portionIIRight = img_width // 2 + per_15_line
        cv2.line(img, (portionIIRight, 0), (portionIIRight, img_height), (50, 0, 100), 2)
        if max_area_mdl_width > portionIIRight:
            section = 'F'
        # right line
        portionIIIRight = img_width // 2 + per_40_line
        cv2.line(img, (portionIIIRight, 0), (portionIIIRight, img_height), (100, 0, 0), 2)
        if max_area_mdl_width > portionIIIRight:
            section = 'G'


        return section


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()
    position = FaceLocation()
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        section = position.drawSection(img, bboxs)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.putText(img, f'Section: {section}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
