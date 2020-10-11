import cv2
import numpy as np
import os
import math
from ex4 import second_deriv_image
os.chdir('C:\\Projects\\ITS8030\\homework1')
"""
Task 5: Image sharpening
"""
def sharpen_image(image : np.ndarray, sigma : float, alpha : float, in_place : bool = False) -> np.ndarray :
    return image - alpha * second_deriv_image(image, sigma)

# Use
img_input = cv2.imread('ex5_input.png', cv2.IMREAD_GRAYSCALE).astype("float32")
img_result = sharpen_image(img_input, 1., 1.).astype("uint8")
cv2.imwrite('ex5_output.png', img_result)
