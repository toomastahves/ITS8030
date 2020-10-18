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
    kernelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    imagex = convolution(image, kernelx)
    kernely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    imagey = convolution(imagex, kernely)

    imagex = imagex / 8
    imagey = imagey / 8

    magnitude = np.hypot(imagex, imagey)
    # Adding 0.01 is hack to avoid division by zero error
    # Multiplying by 100, so image will be visible, not black
    orientation = np.arctan(imagey / (imagex + 0.01)) * 100

    return [magnitude, orientation]

# Use
def run():
    img_input = cv2.imread('input\\ex6_input.jpg', cv2.IMREAD_GRAYSCALE)
    img_result = sobel_image(img_input)
    cv2.imwrite('output/ex6a_output.jpg', img_result[0])
    cv2.imwrite('output/ex6b_output.jpg', img_result[1])

#run()