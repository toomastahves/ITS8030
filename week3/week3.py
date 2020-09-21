import cv2
import numpy as np
import os
os.chdir('C:\\Projects\\ITS8030\\week3')

img = cv2.imread('messi5.jpg', cv2.IMREAD_GRAYSCALE)
imgc = cv2.imread('messi5.jpg', cv2.IMREAD_COLOR)

#cv2.imshow("test",img) 
#cv2.waitKey(0)  
#cv2.destroyAllWindows()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

diff = img2 - img;
img2_int = diff.astype('uint8');

img2_plt = np.array([[[j,j,j] for j in i] for i in img])
import matplotlib.pyplot as plt
plt.imshow(img2_plt)
plt.show()

import scipy
from scipy.signal import convolve2d

meank =  1./9. * np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
identk =  1. * np.array([[0,0,0],[0,1,0],[0,0,0]])

img2 = convolve2d(img, meank, mode="same")
img2_int = img2.astype('uint8')
