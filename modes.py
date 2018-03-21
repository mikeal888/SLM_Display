import numpy as np 
from scipy.special import eval_genlaguerre
from scipy.misc import factorial
import matplotlib.pyplot as plt
from matplotlib import cm
import display_hologram as dh
from win32api import EnumDisplayMonitors
import cv2

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

	def __init__(self, r, phi):
		
		self.state = None
		self.r = r
		self.phi = phi

	def add_mode(self, A, theta, l, p):

		if self.state is None:
			self.state = A*np.exp(1j*theta)*LG_pl(self.r, self.phi, l, p)
		
		else:
			self.state += A*np.exp(1j*theta)*LG_pl(self.r, self.phi, l, p)


	def superposition(self, Params):
		# Parameter array should be = [A, theta, l, p]
		# use create_parameter_array

		for ii in Params:
			self.add_mode(ii[0],ii[1],ii[2],ii[3])


	def add_noise(self, sigma=0.2):
		# add noise 

		self.state = np.real(np.conjugate(self.state)*self.state)*(1+np.random.normal(0,sigma,self.state.shape))


	def intensity(self, noise=False, sigma=0.2):

		profile = np.real(np.conjugate(self.state)*self.state)
		x, y = polar2cartesian(self.r, self.phi)
		cmap = 'gist_heat'
		plt.pcolormesh(x, y, profile, cmap = cmap)
		plt.xlim((np.min(x)/2,np.max(x)/2))
		plt.ylim((np.min(y)/2,np.max(y)/2))
		plt.show()

	def phase(self):

		gradient = np.angle(self.state)
		x, y = polar2cartesian(self.r, self.phi)
		cmap = 'gray'
		plt.pcolormesh(x, y, gradient, cmap=cmap)
		plt.show()


if __name__ == '__main__':

	monitors = EnumDisplayMonitors()
	pxl_x, pxl_y = (monitors[1][2][2]-monitors[1][2][0]), (monitors[1][2][3]-monitors[1][2][1])

	# coordinates
	x_range, y_range = 15, 15
	x0, y0 = 0, 0
	# pxl_x, pxl_y = 800, 600
	z = 0

	x = np.linspace(-x_range+x0,x_range,pxl_x)
	y = np.linspace(-y_range+y0,y_range,pxl_y)

	xx, yy = np.meshgrid(x,y)

	r_mesh, phi_mesh = cartesian2polar(xx,yy)

	# create superposition
	A = np.array([1,0,1])
	theta = np.array([0,0,0])
	l = np.array([-3,0,3])
	p = np.array([0,0,0])

	parameter_array = create_parameter_array(A, theta, l, p)

	state1 = Qstate(r_mesh,phi_mesh)
	state1.superposition(parameter_array)
	phase = np.angle(state1.state)
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

