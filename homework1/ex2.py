import cv2
import numpy as np
import os
import math
os.chdir('C:\\Projects\\ITS8030\\homework1')

"""
Task 2: Gaussian blur

Implement the function

gaussian_blur_image(image : np.ndarray, sigma : float, in_place : bool) -> np.ndarray 

to Gaussian blur an image. "sigma" is the standard deviation of the Gaussian.
Use the function mean_blur_image as a template, create a 2D Gaussian filter
as the kernel and call the convolution function of Task 1.
Normalize the created kernel using the function normalize_kernel() (to
be implemented in a lab) before convolution. For the Gaussian kernel, use
kernel size = 2*radius + 1 (same as the Mean filter) and radius = int(math.ceil(3 * sigma))
and the proper normalizing constant.

To do: Gaussian blur the image "songfestival.jpg" using this function with a sigma of 4.0,
and save as "task2.png".
"""
def gaussian_blur_image(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray :
    radius = int(math.ceil(3 * sigma))
    kernel_size = 2 * radius + 1
    gaussian_filter = np.zeros((kernel_size, kernel_size))
    print(gaussian_filter)
    return image

img_input = cv2.imread('songfestival.jpg', cv2.IMREAD_GRAYSCALE)
img_output = gaussian_blur_image(img_input, 2)
cv2.imwrite('task2.png', img_output)
