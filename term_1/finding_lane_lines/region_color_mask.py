"""region_color_mask.py

Code from lesson example. Modified to complete the excercise
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image and print out some stats
image = mpimg.imread('test.jpg')
print 'This image is: {}'.format(type(image)), \
      'with dimensions: {}'.format(image.shape)

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)
line_image = np.copy(image)

# Define our color selection criteria
rgb_th = [200, 200, 200]
print 'Thresholds: {}'.format(rgb_th)

# Define a triangle region of interest
# Keep in mind the origin (x=0, y=0) is in the upper left in image processing
# Note: if you run this code, you'll find these are not sensible values!!
# But you'll get a chance to play with them soon in a quiz
left_bottom = [100, ysize]
right_bottom = [xsize - 100, ysize]
apex = [xsize / 2, 300]

# Fit lines (y=Ax+B) to identify the  3 sided region of interest
# np.polyfit() returns the coefficients [A, B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit(
    (right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit(
    (left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Use a "bitwise OR" to identify pixels below the threshold
color_thresholds = (image[:, :, 0] < rgb_th[0]) \
    | (image[:, :, 1] < rgb_th[1]) \
    | (image[:, :, 2] < rgb_th[2])

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX * fit_left[0] + fit_left[1])) & \
                    (YY > (XX * fit_right[0] + fit_right[1])) & \
                    (YY < (XX * fit_bottom[0] + fit_bottom[1]))

color_select[color_thresholds] = [0, 0, 0]

# Find where image is both colored right and in the region
line_image[~color_thresholds & region_thresholds] = [255, 0, 0]

# Display the image
plt.imshow(color_select)
plt.show()
plt.imshow(line_image)
plt.show()
