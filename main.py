import numpy as np 
from scipy.special import eval_genlaguerre
from scipy.misc import factorial
import matplotlib.pyplot as plt
from matplotlib import cm
import display_hologram as dh
from win32api import EnumDisplayMonitors
import cv2

# Import functions and objects for slm_essentials
from slm_essentials import *

# Time code
import time
t0 = time.time()


############### ------ SLM parameters --------- #####################

# Get monitors 
monitors = EnumDisplayMonitors()

# Get pixels of second monitor
pxl_x, pxl_y = (monitors[1][2][2]-monitors[1][2][0]), (monitors[1][2][3]-monitors[1][2][1])

# SLM screen dimensions in mm
x_range, y_range = 15.8, 12

# Display one hologram or two?
two_display = False

if two_display is True:
	pxl_x = int(pxl_x/2)
	x_range = x_range/2

# Wavelength of light used.
wavelength = (600e-9)*1e3

################# ----- Encoding Hologram E parameters -------- ################
### All units in mm

# centroid
x0_e, y0_e = 0, 0
z_e = 0

# diffraction grating angles
xrad_e = 2
yrad_e = 0

# focal length of digital lens
focal_e = 10e3   #in mm

# beam radius of input beam
beam_waist_e = 2

# Create blanck array with SLM parameters
x_e = np.linspace(-x_range+x0_e, x_range+x0_e,pxl_x)
y_e = np.linspace(-y_range+y0_e ,y_range+y0_e ,pxl_y)
xx_e, yy_e = np.meshgrid(x_e,y_e)
r_mesh_e, phi_mesh_e = cartesian2polar(xx_e,yy_e)

### CREATE QUANTUM STATE

# create superposition
A_e = np.array([1,0,1])
theta_e = np.array([0,0,0])
l_e = np.array([3,0,-2])
p_e = np.array([0,0,0])

# Create parameter array 
parameter_array_e = create_parameter_array(A_e, theta_e, l_e, p_e)

# Make encoded amplitude matrix
Encoding = Qstate(r_mesh_e,phi_mesh_e, wavelength=wavelength, w0=beam_waist_e)
Encoding.superposition(parameter_array_e)

# Create hologram object of encoded amplitude matrix
hologram_e = QHologram(Encoding)

# Add processing to hologram (digital lens, diffraction grating angle, masking, blazzing)
add_angle_e = True
add_lens_e = False
josa_e = True
blazzing_e = True

if add_angle_e:
	hologram_e.add_angles(xrad_e, yrad_e)

if add_lens_e:
	hologram_e.LensHologram(focal_e)

if josa_e:
	hologram_e.josa_array()

# Create final hologram 
hologram_matrix_e = hologram_e.create_hologram()


################# --------- Measurement Hologram M parameters ------ ##############

# Only calculate measurement arrays if displaying two images (save time)
if two_display:

	x0_m, y0_m = 0, 0
	z_m = 0
	xrad_m = -7
	yrad_m = 0

	focal_m = 10e3 # in mm

	x_m = np.linspace(-x_range+x0_m ,x_range+x0_m ,pxl_x)
	y_m = np.linspace(-y_range+y0_m ,y_range+y0_m ,pxl_y)

	xx_m, yy_m = np.meshgrid(x_m,y_m)

	r_mesh_m, phi_mesh_m = cartesian2polar(xx_m,yy_m)

	beam_waist_m = 2

	A_m = np.array([1,0,0])
	theta_m = np.array([0,0,0])
	l_m = np.array([2,0,0])
	p_m = np.array([0,0,0])

	parameter_array_m = create_parameter_array(A_m, theta_m, l_m, p_m)

	Measurement = Qstate(r_mesh_m,phi_mesh_m, wavelength=wavelength, w0=beam_waist_m)
	Measurement.superposition(parameter_array_m)
	
	hologram_m = QHologram(Measurement)
	
	add_angle_m = True
	add_lens_m = False
	josa_m = True
	blazzing_m = True

	if add_angle_m:
		hologram_m.add_angles(xrad_m, yrad_m)
	
	if add_lens_m:
		hologram_m.LensHologram(focal_m)

	if josa_m:
		hologram_m.josa_array()
	
	hologram_matrix_m = hologram_m.create_hologram()

	# Print all essential information
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

	# Concatenate images next too each other (display side by side)
	hologram = np.concatenate((hologram_matrix_e, hologram_matrix_m), axis=1)

	# Convert image to correct display numbers 0 - 255
	hologram = convert_image(hologram)

	# Add blazzing 
	if blazzing_e and blazzing_m:
		hologram = blazzing(hologram)

	# Finish timing and display hologram on second monitor
	t1 = time.time()
	print("%.2f s to run"%(t1-t0))
	display_image(hologram)

else:
	# Print all essential information
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

	# Convert image to correct display numbers 0 - 255
	hologram_matrix_e = convert_image(hologram_matrix_e)

	# Add blazzing 
	if blazzing_e is True:
		hologram_matrix_e = blazzing(hologram_matrix_e, points=None)

	# Finish timing and display hologram on second monitor
	t1 = time.time()
	print("%.2f s to run"%(t1-t0))
	display_image(hologram_matrix_e)

