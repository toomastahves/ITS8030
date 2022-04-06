import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

# Tutorial https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

i = 11
query_image = 'lusitania_{}.png'.format(i)
train_image = 'lusitania_{}.png'.format(i)
img1 = cv2.imread('images/templates/{}'.format(query_image), 0)
img2 = cv2.imread('images/original/{}'.format(train_image), 0)

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
draw_params = dict(matchColor=(0,255,0),singlePointColor=(255,0,0),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,**draw_params)
#cv2.imwrite('ex3_3_result/result.png', img3)

plt.imshow(img3)
plt.show()
