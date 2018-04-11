import numpy as np 
import matplotlib.pyplot as plt 

def LensPhaseArray(x, wavelength, f):
	# the offset (x-x_offset) is already taken into account by the x array
	return (-np.pi/(f*wavelength))*x**2

def LensPhaseMatrix(x,y,wavelength,f):
	# the offset (x-x_offset) is already taken into account by the x array
	return (-np.pi/(f*wavelength))*((x**2)+(y**2))

def LensHologram(x,y,wavelength,f):
	# the offset (x-x_offset) is already taken into account by the x array
	return np.exp( (-1j*np.pi/(f*wavelength))*((x**2)+(y**2)) )

if __name__ == '__main__':

	x0 = 0
	x_range = 5e-3
	pxl_x = 100
	x = np.linspace(-x_range-x0, x_range-x0, pxl_x)

	y0 = 0
	y_range = 5e-3
	pxl_y = 100
	y = np.linspace(-y_range-y0, y_range-y0, pxl_y)

	xx,yy = np.meshgrid(x,y)

	wavelength = 600e-9
	f = 1

	z = LensHologram(xx, yy, wavelength, f)

	print(np.min(z), np.max(z))

	plt.imshow(z)
	plt.show()
