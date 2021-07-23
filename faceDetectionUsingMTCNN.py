import mtcnn
import cv2


def findFaces(video_capture):
    # Initialize variables
    facesInAFrame = []

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        if ret:  # if video is still left continue creating frames
            detector = mtcnn.MTCNN()
            # detect faces in the image
            faces = detector.detect_faces(frame)
            for face in faces:
                print(face)

            for result in faces:
                x, y, width, height = result['box']
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    # print("total detections: " + str(len(faces)))
    # print(faces)

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


def main():
    video_capture = cv2.VideoCapture("assets/NRGgaming.mp4")
    findFaces(video_capture)


if __name__ == "__main__":
    main()
