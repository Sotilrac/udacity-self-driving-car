"""lane_color_th.py

Code from lesson example. Modified to complete the excercise
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys

# Read in the image and print out some stats
image = mpimg.imread('test.jpg')
print 'This image is: {}'.format(type(image)), \
      'with dimensions: {}'.format(image.shape)

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)

# Define our color selection criteria
rgb_th = [int(th) for th in sys.argv[1:]]
print 'Thresholds: {}'.format(rgb_th)

# Use a "bitwise OR" to identify pixels below the threshold
thresholds = (image[:, :, 0] < rgb_th[0]) \
    | (image[:, :, 1] < rgb_th[1]) \
    | (image[:, :, 2] < rgb_th[2])

color_select[thresholds] = [0, 0, 0]

# Display the image
plt.imshow(color_select)
plt.show()
