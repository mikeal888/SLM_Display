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

import time
t0 = time.time()
## ---- ------ ---- ------ Define Hologram functions first ---------------------- ##
def normalise_image(image):
	image = image/np.sum(image)
	return(image)

def convert_image(image):
	# open cv needs image array to be between [0,255] as uint8 dtype
	image = image - np.min(image)
	image = image/np.max(image)
	image = np.floor((255)*image)
	return image


def display_image(image, on_monitor=2):

	# Get monitors. Assuming that the slm is identified as monitor 2

	monitors = EnumDisplayMonitors()

	assert len(monitors) >= 2, "Less than 2 monitors detected, check display settings"

	x_loc, y_loc = monitors[on_monitor-1][2][0], monitors[on_monitor-1][2][1]

	if image.dtype is not np.dtype('uint8'):
		print('\nNOTE: IMAGE IS NOT UINT8. CONVERTING TO UINT8\n')
		image = np.uint8(image)

	cv2.imshow("image", image)
	cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	cv2.moveWindow("image",x_loc, y_loc)		

def display_two_images(image1,image2,ax=1):

	image = np.concatenate((image1, image2), axis=ax)
	display_image(image)

def blazzing(image, points=None):
	# create blazing function uint8 ==> 258

	if points is None:
		points = np.array([[-np.pi,-np.pi]
			,[-0.5,-np.pi]
			,[0,0]
			,[0.5,np.pi]
			,[np.pi,np.pi],])

	spline = interp1d(points[:,0], points[:,1], kind='cubic')
	colormap_values = np.linspace(-np.pi,np.pi,(2**8), endpoint=True)
	colormap = spline(colormap_values)
	np.place(colormap, colormap>np.pi, np.pi)
	np.place(colormap, colormap<-np.pi, -np.pi)

	colormap = np.floor((255)*(colormap+np.pi)/(2*np.pi))

	r,c = np.shape(image)

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

## - Create parameter array for superposition
def create_parameter_array(A,theta,l,p):
	return(np.transpose(np.append([A],[theta,l,p], axis=0)))

def LensHologram(x,y,wavelength,f):
	# the offset (x-x_offset) is already taken into account by the x array
	return np.exp( (-1j*np.pi/(f*wavelength))*((x**2)+(y**2)) )
	
## ---------------------- Create Quantum state object ------------------------ ##
# This class will be used to generate the holograms of given quantum states based of the 
# parameter array

class Qstate:

	def __init__(self, r, phi, wavelength=0.001, w0=1):
		
		self.state = None
		self.hologram_matrix = None
		self.r = r
		self.phi = phi
		self.x, self.y = polar2cartesian(self.r, self.phi)
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
		cmap = 'gist_heat'
		plt.pcolormesh(self.x, self.y, profile, cmap = cmap)
		plt.xlim((np.min(x)/2,np.max(x)/2))
		plt.ylim((np.min(y)/2,np.max(y)/2))
		plt.show()

	def phase_image(self):

		gradient = np.angle(self.state)
		x, y = polar2cartesian(self.r, self.phi)
		cmap = 'gray'
		plt.pcolormesh(self.x, self.y, gradient, cmap=cmap)
		plt.show()

	def josa_array(self):

		if self.hologram_matrix is None:
			self.hologram_matrix = np.exp(1j*np.abs(self.state)*np.angle(self.state)/(2*np.pi))
		else:
			self.hologram_matrix = np.exp(1j*np.abs(self.hologram_matrix)*np.angle(self.hologram_matrix)/(2*np.pi))

	def add_angles(self, xangle=0, yangle=0):
		# angle in millirad

		if self.hologram_matrix is None:
			self.hologram_matrix = np.exp(1j*(xangle)*self.x)*np.exp(1j*(yangle)*self.y)*self.state
		else:
			self.hologram_matrix = np.exp(1j*(xangle)*self.x)*np.exp(1j*(yangle)*self.y)*self.hologram_matrix

	def reset_hologram(self):
		# reset hologram to initial state
		self.hologram_matrix = self.state

	def create_hologram_matrix(self, xrad=0, yrad=0, add_angle=False, josa=False):

		# This function will create matrix array and erase previous parameters
		# This will allow you to update hologram and erase old parameters

		self.reset_hologram()
		
		if add_angle is True:
			self.add_angles(xrad, yrad)

		if josa is True:
			self.josa_array()

		return np.angle(self.hologram_matrix)




if __name__ == '__main__':

	## ------ SLM parameters --------- ###

	monitors = EnumDisplayMonitors()

	# If displaying two images horizontally, you need to divide pxl_x by 2
	two_display = True
	pxl_x, pxl_y = (monitors[1][2][2]-monitors[1][2][0]), (monitors[1][2][3]-monitors[1][2][1])

	# coordinates in mm
	x_range, y_range = 15.8, 12

	if two_display is True:
		pxl_x = int(pxl_x/2)
		x_range = x_range/2

	wavelength = 810e-9

	## ----- Encoding Hologram E parameters -------- ##
	add_angle_e = True
	josa_e = True
	blazzing_e = True
	
	x0_e, y0_e = 0, 0
	z_e = 0
	xrad_e = 20
	yrad_e = 0

	x_e = np.linspace(-x_range+x0_e, x_range+x0_e,pxl_x)
	y_e = np.linspace(-y_range+y0_e ,y_range+y0_e ,pxl_y)

	xx_e, yy_e = np.meshgrid(x_e,y_e)

	r_mesh_e, phi_mesh_e = cartesian2polar(xx_e,yy_e)

	## Create encoding
	beam_waist_e = 1

	# create superposition
	A_e = np.array([1,0,0])
	theta_e = np.array([0,0,0])
	l_e = np.array([0,0,0])
	p_e = np.array([0,0,0])

	parameter_array_e = create_parameter_array(A_e, theta_e, l_e, p_e)

	Encoding = Qstate(r_mesh_e,phi_mesh_e, wavelength=wavelength, w0=beam_waist_e)
	Encoding.superposition(parameter_array_e)
	hologram_matrix_e = Encoding.create_hologram_matrix(xrad_e,yrad_e,add_angle=add_angle_e,josa=josa_e)
	test = Encoding.create_hologram_matrix(xrad_e,yrad_e,add_angle=add_angle_e,josa=josa_e)
	## --------- Measurement Hologram M parameters ------ #
	add_angle_m = True
	josa_m = True
	blazzing_m = True

	x0_m, y0_m = 0, 0
	z_m = 0
	xrad_m = 20
	yrad_m = 0

	x_m = np.linspace(-x_range+x0_m ,x_range+x0_m ,pxl_x)
	y_m = np.linspace(-y_range+y0_m ,y_range+y0_m ,pxl_y)

	xx_m, yy_m = np.meshgrid(x_m,y_m)

	r_mesh_m, phi_mesh_m = cartesian2polar(xx_m,yy_m)

	beam_waist_m = 1

	A_m = np.array([1,0,1])
	theta_m = np.array([0,0,0])
	l_m = np.array([12,0,-12])
	p_m = np.array([0,0,0])

	parameter_array_m = create_parameter_array(A_m, theta_m, l_m, p_m)

	Measurement = Qstate(r_mesh_m,phi_mesh_m, wavelength=wavelength, w0=beam_waist_m)
	Measurement.superposition(parameter_array_m)
	hologram_matrix_m = Measurement.create_hologram_matrix(xrad_m,yrad_m,add_angle=add_angle_m,josa=josa_m)

	if two_display is True:
		print("\n** ------  Displaying two holograms | E | M | ------ **\n")

		print("## Hologram parameters ##\n")
		print("Hologram dimensions in mm (w, h) = (%s, %s)"%(y_range,x_range))
		print("Number of Pixels per hologram = (%s,%s)"%(pxl_x,pxl_y))

		print("\n## Encoding parameters are: ##\n")
		print("Centroid in mm (x0, y0) = (%s, %s)"%(x0_e,y0_e))
		print("Beam size in mm w0=%s"%(beam_waist_e))

		if add_angle_e:
			print("Angles in millirad (x_angle, y_angle) = (%s, %s)"%(xrad_e,yrad_e))
		else: 
			print("No angle added")

		if josa_e:
			print("Josa intensity masking added")	

		print("\n## Measurement parameters are: ##\n")
		print("Centroid in mm (x0, y0) = (%s, %s)"%(x0_m,y0_m))
		print("Beam size in mm w0=%s"%(beam_waist_m))

		if add_angle_m:
			print("Angles in millirad (x_angle, y_angle) = (%s, %s)"%(xrad_m,yrad_m))
		if josa_m:
			print("Josa intensity masking added")	


		hologram = np.concatenate((hologram_matrix_e, hologram_matrix_m), axis=1)
		hologram = convert_image(hologram)
		# hologram_matrix_e = convert_image(hologram_matrix_e)
		# if blazzing_e is True:
		# 	hologram_matrix_e = blazzing(hologram_matrix_e, points=None)

		# hologram_matrix_m = convert_image(hologram_matrix_m)
		# if blazzing_m is True:
		# 	hologram_matrix_m = blazzing(hologram_matrix_m, points=None)
		if blazzing_e and blazzing_m:
			hologram = blazzing(hologram)

		t1 = time.time()
		print("%.2f s to run"%(t1-t0))
		display_image(hologram)

	else:
		print("diplaying one hologram | E |")

		print("## Hologram parameters ##\n")
		print("Hologram dimensions in mm (w, h) = (%s, %s)"%(y_range,x_range))
		print("Number of Pixels per hologram = (%s,%s)"%(pxl_x,pxl_y))
		print("\n## Encoding parameters are: ##\n")
		print("Centroid in mm (x0, y0) = (%s, %s)"%(x0_e,y0_e))
		if add_angle_e:
			print("Angles in millirad (x_angle, y_angle) = (%s, %s)"%(xrad_e,yrad_e))
		else: 
			print("No angle added")

		hologram_matrix_e = convert_image(hologram_matrix_e)

		if blazzing_e is True:
			hologram_matrix_e = blazzing(hologram_matrix_e, points=None)

		t1 = time.time()
		print("%.2f s to run"%(t1-t0))

		display_image(hologram_matrix_e)


