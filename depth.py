import cv2
import numpy as np

img_disp_name = 'image (2).jpg'
img_disp_name = cv2.cvtColor(img_disp_name, cv2.COLOR_BGR2GRAY)

img_d = cv2.imread(img_disp_name, cv2.IMREAD_UNCHANGED).astype(np.float32)
img_d[img_d > 0] = (img_d[img_d > 0] - 1) / 256

disp = img_d[320, 240]
print(disp)
depth = (0.209313 * 2262.52) / disp
print(depth)
