import numpy as np 
from scipy.special import eval_genlaguerre
from scipy.misc import factorial
import matplotlib.pyplot as plt
from matplotlib import cm
import display_hologram as dh
from win32api import EnumDisplayMonitors
import cv2

import display_hologram as dh

def Gaussian(xdata, x0, sigma):

	const = (1/(sigma*np.sqrt(2*np.pi)))*(np.max(xdata)-np.min(xdata))/len(xdata)
	gauss = const*np.exp(-0.5*((xdata - x0)/sigma)**2) 

	return(gauss)

def Gaussian2d(xdata, ydata, x0=0, y0=0, sigma_x=1, sigma_y=1):

	gauss = Gaussian(xx, x0, sigma_x)*Gaussian(yy,y0,sigma_y)

	return(gauss)

def cartesian2polar(xdata, ydata):

	r = np.sqrt(xdata**2 + ydata**2)
	phi = np.arctan2(ydata, xdata)

	return(r, phi)

def polar2cartesian(r,phi):
	return(r*np.cos(phi), r*np.sin(phi))

def LG_pl(r, phi, l, p, z=0, wavelength=0.001, w0=1):

	# define k vector, rayliegh range, beam waist
	k = 2*np.pi / wavelength
	zR = (k*w0**2)/2
	wz = np.sqrt(2*(z**2 + zR**2)/(k*zR))
	al = abs(l)

	# define all coefficients
	cf1 = np.sqrt((2*factorial(p))/((wz**2)*factorial(al+p)*np.pi))
	cf2 = ((r*np.sqrt(2))/wz)**al
	cf3 = np.exp(-(r**2)/wz**2)
	cf4 = np.exp(1j*l*phi) 
	cf5 = np.exp(1j*(k*z*r**2)/(2*(z+zR**2)))
	cf6 = np.exp(-1j*(2*p + al + 1)*np.arctan(z/zR))

	# Calculate lg mode
	LG = cf1*cf2*cf3*cf4*cf5*cf6*eval_genlaguerre(p, al, (2*r**2)/wz**2)

	return(LG)

# Create parameter array for superposition
def create_parameter_array(A,theta,l,p):
	return(np.transpose(np.append([A],[theta,l,p], axis=0)))


class Qstate:

	def __init__(self, r, phi, wavelength=0.001, w0=1):
		
		self.state = None
		self.r = r
		self.phi = phi
		self.wavelength= wavelength
		self.w0 = w0

	def add_mode(self, A, theta, l, p):

		if self.state is None:
			self.state = A*np.exp(1j*theta)*LG_pl(self.r, self.phi, l, p, wavelength=self.wavelength, w0=self.w0)
		
		else:
			self.state += A*np.exp(1j*theta)*LG_pl(self.r, self.phi, l, p, wavelength=self.wavelength, w0=self.w0)


	def superposition(self, Params):
		# Parameter array should be = [A, theta, l, p]
		# use create_parameter_array

		for ii in Params:
			self.add_mode(ii[0],ii[1],ii[2],ii[3])


	def intensity_image(self, noise=False, sigma=0.2):

		profile = np.real(np.conjugate(self.state)*self.state)
		x, y = polar2cartesian(self.r, self.phi)
		cmap = 'gist_heat'
		plt.pcolormesh(x, y, profile, cmap = cmap)
		plt.xlim((np.min(x)/2,np.max(x)/2))
		plt.ylim((np.min(y)/2,np.max(y)/2))
		plt.show()

	def phase_image(self):

		gradient = np.angle(self.state)
		x, y = polar2cartesian(self.r, self.phi)
		cmap = 'gray'
		plt.pcolormesh(x, y, gradient, cmap=cmap)
		plt.show()

	def josa_array(self):
		return np.exp(1j*np.abs(self.state)*np.angle(self.state)/(2*np.pi))

def normalise_image(image):
	# open cv needs image array to be between [0,1]
	image = image - np.min(image)
	image = image/np.max(image)
	return image


def display_image(image, on_monitor=2):

	# Get monitors. Assuming that the slm is identified as monitor 2

	monitors = EnumDisplayMonitors()

	assert len(monitors) >= 2, "Less than 2 monitors detected, check display settings"

	x_loc, y_loc = monitors[on_monitor-1][2][0], monitors[on_monitor-1][2][1]

	image = normalise_image(image)

	cv2.imshow("image", image)
	cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	cv2.moveWindow("image",x_loc, y_loc)		


if __name__ == '__main__':

	monitors = EnumDisplayMonitors()
	pxl_x, pxl_y = (monitors[1][2][2]-monitors[1][2][0]), (monitors[1][2][3]-monitors[1][2][1])

	# coordinates
	beam_waist = 2
	x_range, y_range = 5, 5
	x0, y0 = 0, 0
	z = 0

	x = np.linspace(-x_range+x0,x_range,pxl_x)
	y = np.linspace(-y_range+y0,y_range,pxl_y)

	xx, yy = np.meshgrid(x,y)

	r_mesh, phi_mesh = cartesian2polar(xx,yy)

	# create superposition
	A = np.array([1,0,1])
	theta = np.array([0,0,0])
	l = np.array([1,0,-1])
	p = np.array([0,0,0])

	parameter_array = create_parameter_array(A, theta, l, p)

	state1 = Qstate(r_mesh,phi_mesh, w0=beam_waist)
	state1.superposition(parameter_array)
	phase = np.angle(state1.josa_array())
	test = np.angle(state1.state)
	# state1.add_noise(sigma=0.000001)


	# freq = np.fft.fft2(state1.state)
	# freq = np.fft.fftshift(freq)
	# freq = np.abs(freq)
	
	# ax1 = plt.subplot(121)
	# ax1.pcolormesh(xx,yy,np.abs(state1.state))
	# ax1.set_xlim([-3,3])
	# ax1.set_ylim([-3,3])
	# ax1.set_title('Spatial')


	# ax2 = plt.subplot(122)
	# ax2.pcolormesh(xx,yy,freq)
	# ax2.set_xlim([-3,3])
	# ax2.set_ylim([-3,3])
	# ax2.set_title('Fourier')
	# plt.show()


