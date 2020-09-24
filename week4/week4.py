from sympy import *
import sympy.plotting as pl

x, y = symbols('x y')
mu = 0
sigma = 1
pi = 3.14
nu = 10

g = Function('g')(x)
g = 1 / (sigma*sqrt(2*pi)) * E**(-1/2*(x-mu)**2/sigma**2)
# plot(g)
res = integrate(g, (x, -2*sigma, 2*sigma)).evalf(10)
print(res)

g2= Function('g2')(x, y)
g2 = 1 / (sigma*sqrt(2*pi)) * E**(-1/2*((x-mu)**2+(y-mu)**2)/sigma**2)
#pl.plot3d(g2)
res2 = integrate(g2, (x, -2*sigma, 2*sigma), (y, -2*sigma, 2*sigma)).evalf(10)
print(res2)

f = Function('f')(x)
f = cos(2*pi*x)

#plot(f, ylim=(-1.2,1.2))
k = symbols('k')

ft = fourier_transform(f,x,k)
pl.plot(ft)

f2 = E**(I*2*pi*nu*x)
