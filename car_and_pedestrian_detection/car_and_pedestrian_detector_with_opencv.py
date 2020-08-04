import cv2

# #image preprocessing for detection
# car_img_file = 'test_images/car.png'
# car_img = cv2.imread(car_img_file)
# gray_car_img = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)

#video preprocessing for detection
motorcycle_dashcam = cv2.VideoCapture('test_images/motorcycle_dashcam.mp4')

#creating car classifier and pedestrian classifier by loading the haar cascade classsifier weights into cv2 cascade classifier
car_classifier_weights = 'car_detector.xml'
pedestrian_classifier_weights = 'pedestrian_detector.xml'
car_classifier = cv2.CascadeClassifier(car_classifier_weights)
pedestrian_classifier = cv2.CascadeClassifier(pedestrian_classifier_weights)

# #detecting cars from images
# cars = car_classifier.detectMultiScale(gray_car_img)
# for (x, y, w, h) in cars:
#     cv2.rectangle(car_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#detecting cars from video
while True:
    read_successful, frame = motorcycle_dashcam.read()
    if(read_successful):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_classifier.detectMultiScale(gray_frame)
        pedestrians = pedestrian_classifier.detectMultiScale(gray_frame)
        #drawing rectangles around cars
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #drawing rectangles around pedestrians
        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    else:
        break

    #displaying the work!!!
    cv2.imshow('car and pedestrian detection display', frame)
    key = cv2.waitKey(1)

    if(key==81 or key==113):
        break

motorcycle_dashcam.release()