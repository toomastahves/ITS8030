import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=[10, 8])
ax  = fig.add_subplot(1, 1, 1, projection='3d')

p1    = np.array([4., 5.,  6.])
p2    = np.array([ 4., 5., 1.])
p3    = np.array([ 4., 1., 6.])

ax.scatter([4.,4.,4.],[5.,5.,6.],[6.,1.,1.])

zaxis=np.array([0.,0.,1.])
I = np.array([[1.,0.,0.],
             [0.,1.,0.],
             [0.,0.,1.]])

R=I + np.sin(np.pi/4)*zaxis + (1- np.cos(np.pi/4))*zaxis**2

np.matmul (R,p1)
np.matmul (R,p2)
np.matmul (R,p3)

ax.scatter([4.,4.,4.,10.,5.,10.],[5.,5.,6.,11.,6.,7.],[6.,1.,1.,12.,2.,12.])

plt.show()