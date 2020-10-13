import cv2
import numpy as np
import os
import math
from ex1 import convolution
os.chdir('C:\\Projects\\ITS8030\\homework1')
"""
Task 6: Edge Detection

Implement 
sobel_image(image : np.ndarray, in_place : bool = False) -> np.ndarray
to compute edge magnitude and orientation information. Convert the image into grayscale.
Use the standard Sobel masks in X and Y directions:
[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] and [[1, 2, 1], [0, 0, 0], [-1, -2, -1]] respectively to compute
the edges. Note that the kernel values sum to 0 in these cases, so you don't need to normalize the
kernels before convolving. Divide the image gradient values by 8 before computing the magnitude and
orientation in order to avoid spurious edges. sobel_image should then display both the magnitude and
orientation of the edges in the image.

To do: Compute Sobel edge magnitude and orientation on "cactus.jpg" and save as "task6.png".
"""
def sobel_image(image):
    kernelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    imagex = convolution(image, kernelx)
    kernely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    imagey = convolution(imagex, kernely)

    imagex = imagex / 8
    imagey = imagey / 8

    magnitude = np.hypot(imagex, imagey)
    direction = np.arctan(imagey / imagex)

    return [magnitude, direction]

# Use
def run():
    img_input = cv2.imread('ex6_input.jpg', cv2.IMREAD_GRAYSCALE)
    img_result = sobel_image(img_input)
    cv2.imwrite('ex6a_output.jpg', img_result[0])
    cv2.imwrite('ex6b_output.jpg', img_result[1])

#run()