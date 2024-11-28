import os

import cv2
import face_recognition

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

imgBackground = cv2.imread("Resources/background.png")
# importing mode images
folderPath = "Resources/Modes"
modePath = os.listdir(folderPath)
imgList = []
# print(modePath)
for path in modePath:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
print(len(imgList))

while True:
    success,img = cam.read()
    imgBackground[162:162+480,55:55+640] = img
    imgBackground[44:44+633,808:808+414] = imgList[0]
    cv2.imshow("Webcam",img)
    cv2.imshow("Face Attendence",imgBackground)
    cv2.waitKey(1)