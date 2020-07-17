import cv2
from random import randrange

#LOADING TRAINED DATA
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# #LOADING IMAGE
# image_path = 'test_images/rdj_peter.jpg'
# img = cv2.imread(image_path)

# #CONVERTING TO GRAYSCALE
# grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #DETECT FACES
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# #DRAW GREEN RECTANGLE[S] AROUND THE PREDICTED FACE[S]
# for face_coordinate in face_coordinates:
#     (x, y, w, h)=face_coordinate
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# #SHOWING IMAGE
# cv2.imshow('RDJ', img)
# cv2.waitKey()

#LOADING WEBCAME STREAM
webcam = cv2.VideoCapture(0)
#PASS VIDEO PATH AS AN ARGUMENT TO ABOVE FUNCTION TO DETECT FACES IN LOCAL VIDEOS

#SAME THING ON WEBCAM
while True:
    is_frame_read_success, frame = webcam.read()
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)
    for face_coordinate in face_coordinates:
        (x, y, w, h)=face_coordinate
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(255), randrange(255), randrange(255)), 2)
    cv2.imshow('WEBCAM', frame)
    key = cv2.waitKey(1)

    if(key==81 or key==113):
        break

webcam.release()