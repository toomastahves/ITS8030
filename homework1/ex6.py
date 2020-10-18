import cv2
import numpy as np
import os
import math
from ex1 import convolution
os.chdir('C:\\Projects\\ITS8030\\homework1')
"""
Task 6: Edge Detection
"""
def sobel_image(image):
    print('Sobel image started..')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    image_x = convolution(image, kernel_x, False)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    image_y = convolution(image, kernel_y, False)

    image_x /= 8
    image_y /= 8

    magnitude = np.hypot(image_x, image_y)
    orientation = np.arctan(image_y / image_x) * 128
    nans = np.isnan(orientation)
    orientation[nans] = 0
    return magnitude, orientation

# Use
def run():
    img_input = cv2.imread('input\\ex6_input.jpg', cv2.IMREAD_COLOR).astype("float32")
    magnitude, orientation = sobel_image(img_input)
    cv2.imwrite('output/ex6a_output.jpg', magnitude)
    cv2.imwrite('output/ex6b_output.jpg', orientation)

#run()