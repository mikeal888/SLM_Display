import numpy as np 
from scipy.special import eval_genlaguerre
from scipy.misc import factorial
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import cm
import display_hologram as dh
from win32api import EnumDisplayMonitors
import cv2

import display_hologram as dh

## ---- ------ ---- ------ Define Hologram functions first ---------------------- ##

def convert_image(image):

	# open cv needs image array to be between [0,255] 
	# converted to uint8 in display_image function

	image = image - np.min(image)
	image = image/np.max(image)
	image = np.floor((255)*image)
	return image

def display_image(image, on_monitor=2):

	# Get monitors. Assuming that the slm is identified as monitor 2

	monitors = EnumDisplayMonitors()

	assert len(monitors) >= 2, "Less than 2 monitors detected, check display settings"

	# pixel location of second monitor

	x_loc, y_loc = monitors[on_monitor-1][2][0], monitors[on_monitor-1][2][1]

	# convert to uint8

	if image.dtype is not np.dtype('uint8'):
		print('\nNOTE: IMAGE IS NOT UINT8. CONVERTING TO UINT8\n')
		image = np.uint8(image)

	# Display image

	cv2.imshow("image", image)
	cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	cv2.moveWindow("image",x_loc, y_loc)		

def display_two_images(image1,image2,ax=1):

	# Display images side by side

	image = np.concatenate((image1, image2), axis=ax)
	display_image(image)

def blazzing(image, points=None):

	# create blazing function uint8 ==> 255

	# points to fit cubic spline. Create blazzing map

	if points is None:
		points = np.array([[-np.pi,-np.pi]
			,[-0.5,-np.pi]
			,[0,0]
			,[0.5,np.pi]
			,[np.pi,np.pi],])

	# fit spline
	spline = interp1d(points[:,0], points[:,1], kind='cubic')
	colormap_values = np.linspace(-np.pi,np.pi,(2**8), endpoint=True)
	colormap = spline(colormap_values)

	# peak out values at -pi and pi
	np.place(colormap, colormap>np.pi, np.pi)
	np.place(colormap, colormap<-np.pi, -np.pi)

	# convert to uint8 magnitude. Don't convert to uint8 yet due to overlapping issue below
	colormap = np.floor((255)*(colormap+np.pi)/(2*np.pi))

	# overlap parameter (avoids overlap between values)
	op = 1000

	for ii in range(len(colormap)):
		image[image == ii] = op*colormap[ii]

	image = image/op

	return(image)

## -------------------------- Define misc functions ----------------------##

def Gaussian(xdata, x0, sigma):

	const = (1/(sigma*np.sqrt(2*np.pi)))*(np.max(xdata)-np.min(xdata))/len(xdata)
	gauss = const*np.exp(-0.5*((xdata - x0)/sigma)**2) 

	return(gauss)

def Gaussian2d(xdata, ydata, x0=0, y0=0, sigma_x=1, sigma_y=1):

	gauss = Gaussian(xx, x0, sigma_x)*Gaussian(yy,y0,sigma_y)

	return(gauss)

def cartesian2polar(xdata, ydata):
	# convert cartesian to polar
	r = np.sqrt(xdata**2 + ydata**2)
	phi = np.arctan2(ydata, xdata)

	return(r, phi)

def polar2cartesian(r,phi):
	# convert polar to cartesian
	return(r*np.cos(phi), r*np.sin(phi))

def LG_pl(r, phi, l, p, z=0, wavelength=0.001, w0=1):
	# create laguerre gauss modes
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

def create_parameter_array(A,theta,l,p):
	## - Create parameter array for superpositions
	return(np.transpose(np.append([A],[theta,l,p], axis=0)))


## ---------------------- Create Quantum and hologram objects ------------------------ ##

# This class will be used to generate the holograms of given quantum states based of the 
# parameter array

class Qstate:

	# Qstate object contains all the required functions to build an amplitude matrix
	# of a quantum state

	def __init__(self, r, phi, wavelength=0.001, w0=1):
		
		self.state = None
		self.hologram_matrix = None
		self.r = r
		self.phi = phi
		self.x, self.y = polar2cartesian(self.r, self.phi)
		self.wavelength= wavelength
		self.w0 = w0

	def add_mode(self, A, theta, l, p):
		# Add an extra Laguerre gauss mode to state

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

		# Display intensity image of state

		profile = np.real(np.conjugate(self.state)*self.state)
		cmap = 'gist_heat'
		plt.pcolormesh(self.x, self.y, profile, cmap = cmap)
		plt.xlim((np.min(x)/2,np.max(x)/2))
		plt.ylim((np.min(y)/2,np.max(y)/2))
		plt.show()

	def phase_image(self):
		# create an image of the phase 
		gradient = np.angle(self.state)
		x, y = polar2cartesian(self.r, self.phi)
		cmap = 'gray'
		plt.pcolormesh(self.x, self.y, gradient, cmap=cmap)
		plt.show()


class QHologram:

	# QHologram object takes Qstate object as input and manipulates the array.
	# This is essential for adding digital lenses, intensity masking (josa), and diffraction 
	# gratings.

	def __init__(self, qstate):
		# Assert qstate is Qstate object

		assert type(qstate) is Qstate, "QHologram requires Qstate object!"

		self.Qstate = qstate
		self.amplitude_matrix = qstate.state

	def josa_array(self):
		# Add josa intensity masking
		self.amplitude_matrix = np.exp(1j*np.abs(self.amplitude_matrix)*np.angle(self.amplitude_matrix)/(2*np.pi))

	def add_angles(self, xangle=0, yangle=0):
		# add diffraction grating in mrad

		self.amplitude_matrix = np.exp(2*np.pi*1j*(xangle)*self.Qstate.x)*np.exp(2*np.pi*1j*(yangle)*self.Qstate.y)*self.amplitude_matrix

	def LensHologram(self, f):
		# Add digital lens. Offset (x-x_offset) is already taken into account by the x array
		self.amplitude_matrix = np.exp( (-1j*np.pi/(f*self.Qstate.wavelength))*((self.Qstate.x**2)+(self.Qstate.y**2)) )*self.amplitude_matrix

	def reset_hologram(self):
		# reset hologram to initial state
		self.amplitude_matrix = self.Qstate.state

	def create_hologram(self):
		# create image to be displayed on hologram
		return np.angle(self.amplitude_matrix)