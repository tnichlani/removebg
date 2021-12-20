import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 360)
cap.set(cv2.CAP_PROP_FPS, 60)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()

imgbg = cv2.imread("Images/1.jpeg")

while True:

    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imgbg, threshold=0.6)

    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    _, imgStacked = fpsReader.update(imgStacked, color=(0, 0, 255))

    cv2.imshow("Image", imgStacked)
    cv2.waitKey(1)







