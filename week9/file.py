
import cv2
import numpy as np
import os
os.chdir('C:\\Projects\\ITS8030\\week9')

img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
img2 = img[::20,::20]

#img1x1=img[0:1024,0:1024]

#img10x10=img[8000:8000+1024,13000:13000+1024]

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures=1500)
keypoints_sift, descriptors = sift.detectAndCompute(img, None)
keypoints_surf, descriptors = surf.detectAndCompute(img, None)
keypoints_orb, descriptors = orb.detectAndCompute(img, None)
keypoints_orb10, descriptors10 = orb.detectAndCompute(img, None)

harris = cv2.cornerHarris(img,2,3,0.04)
harris = cv2.dilate(harris,None)

# Threshold for an optimal value, it may vary depending on the image.
img[harris>0.01*harris.max()]=[255]

img_feat_orb = cv2.drawKeypoints(img, keypoints_orb, None)
#img10x10feat = cv2.drawKeypoints(img10x10, keypoints_orb, None)


scale_percent = 60 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height) 
img2 = cv2.resize(img,dim,cv2.INTER_AREA)
cv2.imshow('tile1',img2)
#cv2.imshow('image',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
