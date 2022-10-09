import cv2
import numpy
face_tracker = cv2.CascadeClassifier('haarcascade_frontal.xml')
smile_tracker = cv2.CascadeClassifier('haarcascade_smile.xml')

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()
    if not successful_frame_read:
        break

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_expression = face_tracker.detectMultiScale(grayscaled_img)
    for (x, y, w, h) in face_expression:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        the_face = frame[y:y+h, x:x+h]

        the_face_gray = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        face_smile = smile_tracker.detectMultiScale(the_face_gray, 1.7, 20)
        '''for (x_, y_, w_, h_) in face_smile:
            cv2.rectangle(frame, (x_, y_), (x_+w_, y_+h_), (0, 255, 255), 2)'''
        if len(face_smile) > 0:
            cv2.putText(frame, 'smile :)', (x, y+h+40),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255))
        cv2.imshow('smile detector', frame)
        key = cv2.waitKey(1)
        if key == 81 or key == 113:
            break
webcam.release()
cv2.destroyAllWindow()
print("complete")
