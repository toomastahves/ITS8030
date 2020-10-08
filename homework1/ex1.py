import cv2
import numpy as np
import os
import math
os.chdir('C:\\Projects\\ITS8030\\homework1')
"""
Task 1: Convolution
"""
def convolution(img_input, kernel, add = False):
    img_output = np.zeros_like(img_input)
    size = kernel.shape[0]
    pad = math.floor(size / 2)
    img_padded = np.pad(img_input, (pad, pad), 'constant')

    for x in range(img_input.shape[0]):
        for y in range(img_input.shape[1]):
            img_output[x, y] = (kernel * img_padded[x: x+size, y: y+size]).sum()
    
    if(add):
        img_input += 128

    return img_output

# Use
img_input = cv2.imread('ex1_input.jpg', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[0,-1,0],[-1,2,-1],[0,-1,0]])
img_result = convolution(img_input, kernel)
cv2.imwrite('ex1_output.jpg', img_result)
