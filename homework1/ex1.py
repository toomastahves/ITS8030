import cv2
import numpy as np
import os
import math

os.chdir('C:\\Projects\\ITS8030\\homework1')

"""
Task 1: Convolution

Implement the function 
convolution(image : np.ndarray, kernel : np.ndarray, kernel_width : int, kernel_height : int, add : bool, in_place:bool) -> np.ndarray
to convolve an image with a kernel of size kernel_height*kernel_width.
Use zero-padding around the borders for simplicity (what other options would there be?).
Here:
    image is a 2D matrix of class double
    kernel is a 2D matrix with dimensions kernel_width and kernel_height
    kernel_width and kernel_height are the width and height of the kernel respectively
(Note: in the general case, they are not equal and may not be always odd, so you have to ensure that they are odd.)
    if add is true, then 128 is added to each pixel for the result to get rid of negatives.
    if in_place is True, then the output image should be a copy of the input image. The default is False,
    i.e. the operations are performed on the input image.
"""
def convolution(img_input, kernel, add, in_place = False):

    img_output = np.zeros_like(img_input)
    size = kernel.shape[0]
    pad = np.short(np.floor(size / 2))
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
img_result = convolution(img_input, kernel, False)
cv2.imwrite('ex1_output.jpg', img_result)
