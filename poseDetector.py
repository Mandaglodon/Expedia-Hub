import cv2
import mediapipe as mp
import time

webcam = cv2.VideoCapture(0)

cTime = 0
pTime = 0

mppose = mp.solutions.pose
Pose = mppose.Pose()
mpDraw = mp.solutions.drawing_utils

while True:
    successful_frame_read, frame = webcam.read()
    RGBwebcam = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #process the RGB color frame
    result = Pose.process(RGBwebcam)
    #print(result.pose_landmarks)
    if result.pose_landmarks:
        mpDraw.draw_landmarks(frame, result.pose_landmarks,
                              mppose.POSE_CONNECTIONS)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)

    cv2.imshow('poseDetect', frame)
    cv2.waitKey(1)
