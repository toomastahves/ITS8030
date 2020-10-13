import cv2
import numpy as np
import os
import math
from ex1 import convolution
os.chdir('C:\\Projects\\ITS8030\\homework1')
"""
Task 2: Gaussian blur
"""
def gaussian_blur_image(img_input, sigma):
    kernel =  gaussian_blur_kernel(sigma)
    img_result = convolution(img_input, kernel)
    return img_result

def gaussian_blur_kernel(sigma):
    print('Generating gaussian kernel...')
    radius = int(math.ceil(3 * sigma))
    kernel_size = 2 * radius + 1
    kernel = np.zeros((kernel_size, kernel_size))

    m = int(kernel.shape[0] / 2)
    n = int(kernel.shape[1] / 2)
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            res1 = 1 / (2*np.pi*sigma**2)
            res2 = np.exp(-(x**2+y**2)/(2*sigma**2))
            kernel[x+m, y+n] = res1 * res2

    #kernel = kernel / kernel.sum()
    return kernel

# Use
def run():
    img_input = cv2.imread('ex2_input.jpg', cv2.IMREAD_GRAYSCALE)
    img_result = gaussian_blur_image(img_input, 4)
    cv2.imwrite('ex2_output.jpg', img_result)

#run()