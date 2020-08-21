import cv2

face_detector = cv2.CascadeClassifier('face_detector.xml')
smile_detector = cv2.CascadeClassifier('smile_detector.xml')

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()
    if not successful_frame_read:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)
        face = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        smiles = smile_detector.detectMultiScale(gray_face, scaleFactor=1.7, minNeighbors=20)
        for (x_smile, y_smile, w_smile, h_smile) in smiles:
            cv2.rectangle(face, (x_smile, y_smile), (x_smile+w_smile, y_smile+h_smile), (50, 50, 200), 3)
        if(len(smiles)>0):
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))


    cv2.imshow('smile detector', frame)
    key = cv2.waitKey(1)


    if(key==81 or key==113):
        break


webcam.release()
cv2.destroyAllWindows()