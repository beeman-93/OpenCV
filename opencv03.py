# Import dependencies
import cv2
import mediapipe as mp
import time

# Face detection
mp_facedetecor = mp.solutions.face_detection
# Drawing utilities for bounding boxes
mp_draw = mp.solutions.drawing_utils

#Read a video
# I tested with 4 videos "CaiDanmeng (1).mp4", "Man - 76888.mp4", "Business - 136262.mp4", and "Alley - 39837.mp4"
# I put the .py file and videos under the same folder. Otherwise, please state the path.
cap = cv2.VideoCapture("Alley - 39837.mp4")

#Set up a confidence to tell whether we are detecting a face
with mp_facedetecor.FaceDetection(20) as face_detection:
    while True:
        success, image = cap.read()


        start = time.time()

        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Process the image and locate faces
        results = face_detection.process(image)
        # Convert to BGR so the image can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for id, detection in enumerate(results.detections):
                mp_draw.draw_detection(image, detection)
                # Draw bounding boxes
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = image.shape
                # Draw bounding boxes on image frame
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                # Draw the confidence score on top of the boxes
                cv2.rectangle(image, bbox, (255, 0, 255), 2)
                cv2.putText(image,  f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)

        end = time.time()
        totalTime = end - start
        fps = 1/totalTime
        print ("FPS: ", fps)
        cv2.putText(image, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 2)
        cv2.imshow("Find Faces", image)
        if ord('q') == cv2.waitKey(1):
            break