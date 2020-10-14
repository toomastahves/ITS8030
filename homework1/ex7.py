import cv2
import numpy as np
import os
import math
os.chdir('C:\\Projects\\ITS8030\\homework1')
"""
Task 7: Bilinear Interpolation

Implement the function
bilinear_interpolation(image : np.ndarray, x : float, y : float) -> np.ndarray

to compute the linearly interpolated pixel value at the point (x,y) using bilinear interpolation.
Both x and y are real values. Put the red, green, and blue interpolated results in the vector "rgb".

To do: The function rotate_image will be implemented in a lab and it uses bilinear_interpolation
to rotate an image. Rotate the image "yosemite.png" by 20 degrees and save as "task7.png".
"""
def bilinear_interpolation(image, x, y):
    "Returns a  vector containing interpolated red green and blue values (a vector of length 3)"
    "implement the function here"
    raise "not implemented yet!"

def rotate_image (image : np.ndarray, rotation_angle : float, in_place : bool = False) -> np.ndarray :
    rgb = bilinear_interpolation (image, 0.5, 0.0)
    "To be implemented by the lecturer"
    raise "not implemented yet"
