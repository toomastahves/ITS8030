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
def separable_gaussian_blur_image(image, sigma, add):
    kernel = gaussian_blur_kernel(sigma)

    kernel_x = kernel[[math.floor(kernel.shape[1] / 2)],:]
    kernel_y = kernel[:,[math.floor(kernel.shape[0] / 2)]]

    image1 = convolution(image, kernel_x, add)
    image2 = convolution(image1, kernel_y, add)

    return image2

# Use
img_input = cv2.imread('ex3_input.jpg', cv2.IMREAD_GRAYSCALE)
img_result = separable_gaussian_blur_image(img_input, 4, True)
cv2.imwrite('ex3_output.jpg', img_result)
