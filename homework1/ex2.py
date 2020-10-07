import cv2
import numpy as np
import os
import math
from ex1 import convolution

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
def gaussian_blur_kernel(sigma):
    radius = int(math.ceil(3 * sigma))
    kernel_size = 2 * radius + 1
    gaussian_filter = np.zeros((kernel_size, kernel_size))

    m = int(gaussian_filter.shape[0] / 2)
    n = int(gaussian_filter.shape[1] / 2)

    for x in range(-m, m+1):
        for y in range(-n, n+1):
            res1 = 1 / (2*np.pi*sigma**2)
            res2 = np.exp(-(x**2+y**2)/(2*sigma**2))
            gaussian_filter[x+m, y+n] = res1 * res2

    return gaussian_filter


# Use
img_input = cv2.imread('ex2_input.jpg', cv2.IMREAD_GRAYSCALE)
kernel = gaussian_blur_kernel(4)
img_result = convolution(img_input, kernel, False)
cv2.imwrite('ex2_output.jpg', img_result)
