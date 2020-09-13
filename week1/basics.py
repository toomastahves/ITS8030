import cv2
import numpy as np

img = cv2.imread('week1/flower.jpeg')

b,g,r = cv2.split(img)
img = cv2.merge((b,g,r))
img[:,:,2] = 0

cv2.imshow("OpenCV Image Reading", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
