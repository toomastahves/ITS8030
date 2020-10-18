import cv2
import numpy as np
import os
import math
from ex6 import sobel_image
from ex7 import bilinear_interpolation
os.chdir('C:\\Projects\\ITS8030\\homework1')
"""
Task 8: Finding edge peaks
"""
def find_peaks_image(image_in, threshold):
    magnitude, orientation = sobel_image(image_in)
    image_width = image_in.shape[0]
    image_height = image_in.shape[1]
    image_out = np.zeros_like(image_in)

    for x in range(0, image_width):
        for y in range(0, image_height):
            theta = orientation[x][y]

            e1x = x + np.cos(theta)
            e1y = y + np.sin(theta)
            e2x = x - np.cos(theta)
            e2y = y - np.sin(theta)

            e1 = bilinear_interpolation(magnitude, e1x, e1y)
            e2 = bilinear_interpolation(magnitude, e2x, e2y)
            e = magnitude[x][y]

            if e >= e1 and e >= e2 and e > threshold:
                image_out[x][y] = 255

    return image_out

def run():
    img_input = cv2.imread('input\\ex8_input.jpg', cv2.IMREAD_COLOR).astype("float32")
    img_result = find_peaks_image(img_input, 40).astype("uint8")
    cv2.imwrite('output/ex8_output.jpg', img_result)

run()