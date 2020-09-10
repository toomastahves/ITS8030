import cv2

img = cv2.imread('week1/flower.jpeg', 0)
cv2.imshow("OpenCV Image Reading", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
