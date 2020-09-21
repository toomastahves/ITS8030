import cv2
import numpy as np
import os
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

Write a general convolution function that can handle all possible cases as mentioned above.
You can get help from the convolution part of the function mean_blur_image (to be implemented in a lab)
to write this function.
"""
def convolution(img_input, kernel):
    print("start")
    img_output = np.zeros_like(img_input)
    img_padded = np.zeros((img_input.shape[0] + 2, img_input.shape[1] + 2))
    img_padded[1:-1, 1:-1] = img_input

    for x in range(img_input.shape[1]):
        for y in range(img_input.shape[0]):
            img_output[y, x] = (kernel * img_padded[y: y+3, x: x+3]).sum()
    

    print("end")
    return img_output


img_input = cv2.imread('messi5.jpg', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[0,-1,0],[-1,10,-1],[0,-1,0]])

img_sharp = convolution(img_input, kernel)
cv2.imwrite('output.jpg', img_sharp)