import cv2
import mediapipe as mp
import time

webcam = cv2.VideoCapture(0)
mphands = mp.solutions.hands
Hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils
cTime = 0
pTime = 0
while True:
    successful_frame_read, frame = webcam.read()
    RGBwebcam = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = Hands.process(RGBwebcam)
    #print(result.multi_hand_landmarks)
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
            mpDraw.draw_landmarks(frame, handLms, mphands.HAND_CONNECTIONS)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 2)
    cv2.imshow('hand', frame)
    cv2.waitKey(1)
