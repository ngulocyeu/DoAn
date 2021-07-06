import cv2
import os


count = 0
while True:
    img = cv2.imread("Data/Frame/frame%d.jpg" % count)
    count += 1

    print(img)
    cv2.waitKey(1000)
