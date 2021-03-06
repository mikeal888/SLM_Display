import numpy as np 
from scipy.special import eval_genlaguerre, eval_hermite
from scipy.misc import factorial
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import cm
from win32api import EnumDisplayMonitors
import cv2

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

def kill_image():
	# destroy all windows 
	cv2.destroyAllWindows()
## -------------------------- Define misc functions ----------------------##

def Gaussian(xdata, x0, sigma):

	const = (1/(sigma*np.sqrt(2*np.pi)))*(np.max(xdata)-np.min(xdata))/len(xdata)
	gauss = const*np.exp(-0.5*((xdata - x0)/sigma)**2) 

	return(gauss)

def Gaussian2d(xdata, ydata, x0=0, y0=0, sigma_x=1, sigma_y=1):

	gauss = Gaussian(xdata, x0, sigma_x)*Gaussian(ydata,y0,sigma_y)

	return(gauss)

def RadGauss4d(r,r0=0,sigma=1):

	return np.exp(-0.5*((r - r0)/sigma)**4) 

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
	cf1 = np.sqrt((2*factorial(p))/(factorial(al+p)*np.pi))	
	cf2 = ((r*np.sqrt(2))/wz)**al
	cf3 = np.exp(-(r**2)/wz**2)
	cf4 = np.exp(1j*l*phi) 
	cf5 = np.exp(1j*(k*z*r**2)/(2*(z**2+zR**2)))
	cf6 = np.exp(-1j*(2*p + al + 1)*np.arctan(z/zR))

	# Calculate lg mode
	LG = cf1*cf2*cf3*cf4*cf5*cf6*eval_genlaguerre(p, al, (2*r**2)/wz**2)

	return(LG)

def HG_nm(x, y, n, m, z=0, wavelength=0.001, w0=1):
	# Create Hermite Gauss modes
	k = 2*np.pi / wavelength
	zR = (k*w0**2)/2
	wz = np.sqrt(2*(z**2 + zR**2)/(k*zR))

	# Define all coefficients
	cf1 = np.sqrt(2/(np.pi*factorial(n)*factorial(m)))*2**(-(n+m)/2)
	cf2 = w0/wz
	cf3 = np.exp(-(x**2 + y**2)/(wz**2))
	cf4 = np.exp(-1j*(k*z*(x**2 + y**2))/(2*(z**2 + zR**2)))
	cf5 = np.exp(-1j*(n+m+1)*np.arctan(z/zR))

	# make hermite modes
	Hn = eval_hermite(int(n), np.sqrt(2)*x/wz)
	Hm = eval_hermite(int(m), np.sqrt(2)*y/wz)

	# Calculate HG mode
	HG = cf1*cf2*cf3*cf4*cf5*Hn*Hm

	return(HG)

def create_parameter_array(A,theta,alpha,beta):
	## - Create parameter array for superpositions
	return(np.transpose(np.append([A],[theta,alpha,beta], axis=0)))


## ---------------------- Create Quantum and hologram objects ------------------------ ##

# This class will be used to generate the holograms of given quantum states based of the 
# parameter array

class Qstate:

	# Qstate object contains all the required functions to build an amplitude matrix
	# of a quantum state

	def __init__(self, r, phi, wavelength=0.001, w0=1, basis="LG"):
		
		# Initial parameters
		self.state = None
		self.hologram_matrix = None
		self.r = r
		self.phi = phi
		self.x, self.y = polar2cartesian(self.r, self.phi)
		self.wavelength= wavelength
		self.w0 = w0
		self.basis = basis

	def add_mode(self, A, theta, alpha, beta):
		# Add an extra Laguerre gauss mode to state
		# If LG ==> alpha, beta = l, p
		# If HG ==> alpha, beta = n, m

		if self.basis is 'LG':
			if self.state is None:
				self.state = A*np.exp(1j*theta)*LG_pl(self.r, self.phi, alpha, beta, wavelength=self.wavelength, w0=self.w0)
			
			else:
				self.state += A*np.exp(1j*theta)*LG_pl(self.r, self.phi, alpha, beta, wavelength=self.wavelength, w0=self.w0)

		elif self.basis is 'HG':
			if self.state is None:
				self.state = A*np.exp(1j*theta)*HG_nm(self.x, self.y, alpha, beta, wavelength=self.wavelength, w0=self.w0)
			
			else:
				self.state += A*np.exp(1j*theta)*HG_nm(self.x, self.y, alpha, beta, wavelength=self.wavelength, w0=self.w0)

		else:
			print("WARNING! mode parameter ins 'add_mode' must eqaul 'LG' or 'HG'")



	def superposition(self, Params):
		# Parameter array should be = [A, theta, l, p]
		# use create_parameter_array

		for ii in Params:
			self.add_mode(ii[0],ii[1],ii[2],ii[3])

	def propagate(self,d):
		# Propagate image in paraxial limit 
		# This is all based of Labviews function 
		
		# Firstly have to calculate k_z in paraxial limit k_z = k - (k_r^2)/(2k)
		self.k = 2*np.pi/self.wavelength

		def waveNumberArray(N, s):
			# Calculate wavenumber array k_i, 
			# s = physical size of array
			# N = length of array

			nyq = N/2		# Nyquist
			N_vec = np.linspace(0,N-1,N)
			k_vec = 2*np.pi*(((N_vec+nyq)%N)-nyq)/s

			return k_vec

		Ny,Nx = np.shape(self.r) 
		sx, sy = 2*np.abs(self.x[:,0][0]), 2*np.abs(self.y[0][0])
		k_x, k_y= np.meshgrid(waveNumberArray(Nx, sx), waveNumberArray(Ny, sy))

		# Compute k_z
		k_z = self.k - (k_x**2 + k_y**2)/(2*self.k)

		# Convert amplitude matrix to K space for propagation
		fft2d = np.fft.fft2(self.state)

		# Propagate beam using exp(i*k_z*d)
		fft2d = np.exp(1j*k_z*d)*fft2d

		# Convert back to real space

		self.state = np.fft.ifft2(fft2d)


	def intensity_image(self, noise=False, sigma=0.2):

		# Display intensity image of state

		profile = np.real(np.conjugate(self.state)*self.state)
		cmap = 'gist_heat'
		plt.pcolormesh(self.x, self.y, profile, cmap = cmap)
		plt.xlim((np.min(self.x)/2,np.max(self.x)/2))
		plt.ylim((np.min(self.y)/2,np.max(self.y)/2))
		plt.show()

	def phase_image(self):

		# create an image of the phase

		gradient = np.angle(self.state)
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