import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os

# Tutorial https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html

image_path = 'images/original/lusitania/'
images = os.listdir(image_path)


result_path = 'ex3_result_2/'

for i in range(0, len(images)):
    # Load image
    image_name = images[i]
    img = cv.imread(image_path + image_name)
    

    # Use template matching
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)
    
    img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    
    # Save image to file
    image = Image.fromarray(img)
    image.save(result_path + image_name)
