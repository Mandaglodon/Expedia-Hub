import cv2
import mediapipe as mp
import time

webcam = cv2.VideoCapture(0)
cTime = 0
pTime = 0

mpFace = mp.solutions.face_detection
Face = mpFace.FaceDetection()
mpDraw = mp.solutions.drawing_utils

while True:
    successful_frame, img = webcam.read()
    RGBwebca = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Face.process(RGBwebca)

    #print(results)
    if results.detections:
        for id, detection in enumerate(results.detections):
            print(id, detection)
            #mpDraw.draw_detection(img, detection)
            #print(detection.location_data.RELATIVE_BOUNDING_BOX)
            bboxC = detection.location_data.RELATIVE_BOUNDING_BOX
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih),\
                int(bboxC.width*iw), int(bboxC.height*ih)
            cv2.rectangle(img, bbox, (255, 155, 100), 3)

    cv2.imshow('face detector', img)
    cv2.waitKey(1)
