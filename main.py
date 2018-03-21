import numpy as np 
from scipy.special import eval_genlaguerre
from scipy.misc import factorial
import matplotlib.pyplot as plt
from matplotlib import cm
import display_hologram as dh
from win32api import EnumDisplayMonitors
import cv2

# import local files
import modes as md 
import display_hologram as dh 

## Define input parameters
# Get monitors and number of pixels
 
monitors = EnumDisplayMonitors()
pxl_x, pxl_y = (monitors[1][2][2]-monitors[1][2][0]), (monitors[1][2][3]-monitors[1][2][1])