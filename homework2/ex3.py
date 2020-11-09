import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os

image_path = 'homework2/images_original/'
images = os.listdir(image_path)

template_path = 'homework2/templates/breathingHole/'
templates = os.listdir(template_path)

result_path = 'homework2/ex3_result/'

for i in range(0, len(images)):
    # Load image and template
    image_name = images[i]
    image = cv2.imread(image_path + image_name, 0)
    template = cv2.imread(template_path + image_name, 0)
    w, h = template.shape[::-1]

    # Use template matching
    res = cv2.matchTemplate(image, template, cv2.TM_CCORR)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Draw boxes for min and max results
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(image, top_left, bottom_right, 0, 5)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(image, top_left, bottom_right, 0, 5)

    # Save image to file
    image = Image.fromarray(image)
    image.save(result_path + image_name)
