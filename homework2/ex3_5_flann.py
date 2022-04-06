import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Tutorials 
# https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
# https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html

MIN_MATCH_COUNT = 10

i = 11
query_image = 'lusitania_{}.png'.format(i)
train_image = 'lusitania_{}.png'.format(i)
img1 = cv2.imread('images/templates/{}'.format(query_image), 0)
img2 = cv2.imread('images/original/{}'.format(train_image), 0)

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

# Store all the good matches as per Lowe's ratio test
good = []
for i,(m,n) in enumerate(matches):
    # ratio test as per Lowe's paper
    if m.distance < 0.9*n.distance:
        good.append(m)

# Use homography to remove outliers
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

# Draw images
draw_params = dict(matchColor = (0,255,0), singlePointColor = None,matchesMask = matchesMask,flags = 2)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
#cv.imwrite('ex3_5_result/result.png', img3)
plt.imshow(img3)
plt.show()