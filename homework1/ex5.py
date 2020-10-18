import cv2
import numpy as np
import os
import math
from ex4 import second_deriv_image
os.chdir('C:\\Projects\\ITS8030\\homework1')
"""
Task 5: Image sharpening
"""
def sharpen_image(image, sigma, alpha):
    return image - alpha * second_deriv_image(image, sigma)

# Use
def run():
    img_input = cv2.imread('input\\ex5_input.png', cv2.IMREAD_GRAYSCALE)
    img_result = sharpen_image(img_input, 1, 5)
    cv2.imwrite('output/ex5_output.png', img_result)

#run()