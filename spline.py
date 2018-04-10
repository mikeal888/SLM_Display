import numpy as np 
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 

pi = np.pi

points = np.array([[-pi,-pi]
	,[-0.2,-pi]
	,[0,0]
	,[0.2,pi]
	,[pi,pi],])


spline = interp1d(points[:,0], points[:,1],kind='cubic')
x = np.linspace(-pi,pi,(2**8), endpoint=True)
y = spline(x)
np.place(y, y>pi, pi)
np.place(y, y<-pi, -pi)

y = np.uint8((255)*(y+pi)/(2*pi))

def apply_cmap(array, cmap):
	r,c = np.shape(array)

	for ii in range(r):
		for jj in range(c):
			array[ii,jj] = cmap[array[ii,jj]-1]

	return(array)

plt.plot(points[:,0], 255*(points[:,1]+pi)/(2*pi), 'bo', x, y, 'r--')
plt.show()



