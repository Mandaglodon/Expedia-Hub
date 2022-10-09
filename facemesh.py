import cv2
import mediapipe as mp
import time

webcam = cv2.VideoCapture(0)

cTime = 0
pTime = 0
mpFmesh = mp.solutions.face_mesh
Fmesh = mpFmesh.FaceMesh(max_num_faces=2)
mpDraw = mp.solutions.drawing_utils
drawing_spec = mpDraw.DrawingSpec(
    ([0, 255, 0]), thickness=1, circle_radius=2,)
while True:
    success, frame = webcam.read()
    RGBwebcam = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = Fmesh.process(RGBwebcam)

    if results.multi_face_landmarks:
        for faces in results.multi_face_landmarks:
            mpDraw.draw_landmarks(
                frame, faces, mpFmesh.FACEMESH_CONTOURS, drawing_spec, drawing_spec)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)
    cv2.imshow('face mesh', frame)
    cv2.waitKey(1)
