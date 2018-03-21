import cv2
from win32api import EnumDisplayMonitors

# Display hologram in full screen on windows only

def display_image(image, on_monitor=2):

	# Get monitors. Assuming that the slm is identified as monitor 2

	monitors = EnumDisplayMonitors()

	assert len(monitors) >= 2, "Less than 2 monitors detected, check display settings"

	x_loc, y_loc = monitors[on_monitor-1][2][0], monitors[on_monitor-1][2][1]

	cv2.imshow("image", image)
	cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	cv2.moveWindow("image",x_loc, y_loc)	