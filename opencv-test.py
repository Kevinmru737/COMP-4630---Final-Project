#https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
#https://www.geeksforgeeks.org/python/detect-an-object-with-opencv-python/
#https://github.com/Balaje/OpenCV/blob/master/haarcascades/hand.xml
# https://github.com/Aravindlivewire/Opencv/blob/master/haarcascade/aGest.xml
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# 1. Initialize the camera (0 is default, use 1 for external USB cams)
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

stop_cascade = cv.CascadeClassifier('aGest.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    found = stop_cascade.detectMultiScale(gray, minSize=(20, 20))

   
    # Display the resulting frame
    for (x, y, w, h) in found:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()