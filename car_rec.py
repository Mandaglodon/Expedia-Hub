import cv2

trained_car_data = cv2.CascadeClassifier('Car_detector.xml')
webcam = cv2.imread('cars.png')

grayscaled_img = cv2.cvtColor(webcam, cv2.COLOR_BGR2GRAY)

#successful_frame_read, frame = webcam.read()
car_Coordinates = trained_car_data.detectMultiScale(grayscaled_img)
print(car_Coordinates)

for (x, y, w, h) in car_Coordinates:
    cv2.rectangle(webcam, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow('car detector', webcam)
cv2.waitKey()
print('compkete')
