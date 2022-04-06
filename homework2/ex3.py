import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os

# Tutorial https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html

image_path = 'images/original/'
images = os.listdir(image_path)

template_path = 'images/templates/'
templates = os.listdir(template_path)
methods = [ 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
]
result_path = 'ex3_cv2_tm_results/'

for i in images:
    # Load image and template
    image_name = i
    image = cv2.imread(image_path + image_name)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    template = cv2.imread(template_path + image_name, 0)
    w, h = template.shape[::-1]
    for meth in methods:
        method = eval(meth)
        # Use template matching
        res = cv2.matchTemplate(img_gray, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # Draw boxes for min and max results
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img_gray, top_left, bottom_right, 0, 5)
        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img_gray, top_left, bottom_right, 0, 5)
        # Save image to file
        image = Image.fromarray(img_gray)
        image.save(result_path + meth + image_name)
