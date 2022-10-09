import cv2

trained_car_data = cv2.CascadeClassifier('Car_detector.xml')
pedestrian_data = cv2.CascadeClassifier('Car_rec_haarcascade_full body.xml')
webcam = cv2.VideoCapture('videoplayback.mp4')
while True:
    (successful_frame_read, frame) = webcam.read()
    if successful_frame_read:
        grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    #successful_frame_read, frame = webcam.read()
    car_Coordinates = trained_car_data.detectMultiScale(grayscaled_img)
    print(car_Coordinates)
    pedestrian_coordinates = pedestrian_data.detectMultiScale(grayscaled_img)
    print(pedestrian_coordinates)

    for (x, y, w, h) in car_Coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
    for (x, y, w, h) in pedestrian_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('car and pedestrian detector', frame)
    key = cv2.waitKey(1)
    #stop the proccess
webcam.release()
print('complete')
