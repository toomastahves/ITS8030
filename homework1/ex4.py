import cv2
import numpy as np
import os
import math
from ex1 import convolution
from ex2 import gaussian_blur_image
os.chdir('C:\\Projects\\ITS8030\\homework1')
"""
Task 4: Image derivatives
"""
def first_deriv_image_x(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray :
    kernel = np.array([-1, 0, 1])
    image_out = convolution(image, kernel)
    image_out = gaussian_blur_image(image, sigma)
    image_out += 128
    return image_out

def first_deriv_image_y(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray :
    kernel = np.array([[-1], [0], [1]])
    image_out = convolution(image, kernel)
    image_out = gaussian_blur_image(image, sigma)
    image_out += 128
    return image_out

def second_deriv_image(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray :
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]] )
    image_out = convolution(image, kernel)
    image_out = gaussian_blur_image(image, sigma)
    image_out += 128
    return image_out

# Use
#img_input = cv2.imread('ex4_input.jpg', cv2.IMREAD_GRAYSCALE).astype("float32")
#img_result = first_deriv_image_x(img_input, 4)
#cv2.imwrite('ex4a_output.jpg', img_result)
#img_result = first_deriv_image_x(img_input, 4)
#cv2.imwrite('ex4b_output.jpg', img_result)
#img_result = first_deriv_image_x(img_input, 4)
#cv2.imwrite('ex4c_output.jpg', img_result)
