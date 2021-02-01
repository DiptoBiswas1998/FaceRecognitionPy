import cv2
from random import randrange

# Loading pre-trained face recognition data
trained_face_data = cv2.CascadeClassifier('E:\Codes\Python projects\Python project 5 - Face recognition app\haarcascade_frontalface_default.xml')

# Choosing an image
# imgPath = input("Enter image path : ")
# img = cv2.imread(imgPath)

webcam = cv2.VideoCapture(0)

# Iterate over frames
while True :
    # Reading the current frame
    successful_frame_read, frame = webcam.read()
    # Converting to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    # Drawing rectangles around faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)

    cv2.imshow('My Face Recognition App', frame)
    key = cv2.waitKey(1)

    # Press esc to esc
    if key == 27 :
        break

webcam.release()