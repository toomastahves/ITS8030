import cv2
import numpy as np
import os
import math
os.chdir('C:\\Projects\\ITS8030\\homework1')
"""
Task 7: Bilinear Interpolation
"""
def bilinear_interpolation(image, x, y):
    if x < 0 or x > image.shape[1] - 1 or y < 0 or y > image.shape[0] - 1:
        if(len(image.shape) == 3):
            return (255, 255, 255)
        else:
            return 0
    
    y0 = math.floor(y)
    x0 = math.floor(x)
    x1 = math.ceil(x)
    y1 = math.ceil(y)

    q11 = image[y0][x0]
    q12 = image[y0][x1]
    q21 = image[y1][x0]
    q22 = image[y1][x1]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*q11 + wb*q12 + wc*q21 + wd*q22

def rotate_image(interpolation_fn, image_in, rotation_angle):
    radians = math.radians(rotation_angle)
    image_out = np.zeros_like(image_in)
    image_height = image_in.shape[0]
    image_width = image_in.shape[1]
    for y in range(image_height):
        for x in range(image_width):
            x0 = x - image_width / 2.0
            y0 = y - image_height / 2.0
            x1 = x0 * math.cos(radians) - y0 * math.sin(radians)
            y1 = x0 * math.sin(radians) + y0 * math.cos(radians)
            x1 += image_width / 2.0
            y1 += image_height / 2.0
            rgb = interpolation_fn(image_in, x1, y1)
            image_out[y][x] = rgb
    return image_out

# Use
def run():
    img_input = cv2.imread('input\\ex7_input.png', cv2.IMREAD_COLOR)
    img_result = rotate_image(bilinear_interpolation, img_input, 20)
    cv2.imwrite('output/ex7_output.png', img_result)

#run()