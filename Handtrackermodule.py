import cv2
import mediapipe as mp
import time


class Handdetector():
    def __init__(self, mode=False, maxHand=2, Complexity=1, min_detectcon=0.5, min_trackingcon=0.5):
        self.mode = mode
        self.maxHand = maxHand
        self.Complexity = Complexity
        self.min_detectcon = min_detectcon
        self.min_trackingcon = min_trackingcon

        self.mphands = mp.solutions.hands
        self.Hands = self.mphands.Hands(
            self.mode, self.maxHand, self.Complexity, self.min_detectcon, self.min_trackingcon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):

        RGBwebcam = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.Hands.process(RGBwebcam)
        #print(result.multi_hand_landmarks)
        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        frame, handLms, self.mphands.HAND_CONNECTIONS)
        return frame

    def findposition(self, frame, handnum=0, draw=True):
        lmlist = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handnum]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id, cx, cy)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
        return lmlist


def main():
    cTime = 0
    pTime = 0
    webcam = cv2.VideoCapture(0)
    detector = Handdetector()
    while True:
        successful_frame_read, frame = webcam.read()
        detector.findHands(frame)

        list = detector.findposition(frame)
        if len(list) != 0:
            print(list[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(frame, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 2)
        cv2.imshow('hand', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
