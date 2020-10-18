import cv2
import numpy as np
import os
import math
from ex1 import convolution
from ex2 import gaussian_blur_kernel
os.chdir('C:\\Projects\\ITS8030\\homework1')
"""
Task 3: Separable Gaussian blur
"""
def separable_gaussian_blur_image(image, sigma):
    kernel = gaussian_blur_kernel(sigma)

    image1 = convolution(image, kernel[:,[math.floor(kernel.shape[0] / 2)]])
    image2 = convolution(image1, kernel[[math.floor(kernel.shape[1] / 2)],:])

    #image2 += 128;

    return image2

# Use
def run():
    img_input = cv2.imread('input\\ex3_input.jpg', cv2.IMREAD_GRAYSCALE).astype("float32")
    img_result = separable_gaussian_blur_image(img_input, 4)
    cv2.imwrite('output/ex3_output.jpg', img_result)

#run()