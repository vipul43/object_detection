import cv2

webcam = cv2.VideoCapture(0)
eye_classifier_weights = 'eye_detector.xml'
eye_classifier = cv2.CascadeClassifier(eye_classifier_weights)

while True:
    read_successful, frame = webcam.read()
    if(read_successful):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_classifier.detectMultiScale(gray_frame)
        #drawing rectangles around eyes
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        break
    cv2.imshow('eye classifier', frame)
    key = cv2.waitKey(1)

    if(key==81 or key==113):
        break

webcam.release()