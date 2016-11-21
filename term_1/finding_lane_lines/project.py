"""project.py

Test code for the Lesson project. Requires Python 3
"""

import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending
    # on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill
    # color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  

    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    # separate the coordinates between negative and positive slopes
    slopes = {'neg': [],
              'pos': []}
    coords = {'neg': [],
              'pos': []}
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                cat = 'neg'
            else:
                cat = 'pos'
            slopes[cat].append(slope)
            coords[cat].append((x1, y1))
            coords[cat].append((x2, y2))

    def filter_outliers(data, m=2):
        """helper fct to filter outliers"""
        return data[abs(data - np.mean(data)) < m * np.std(data)]

    # Find average slopes for both categories and line extremes
    for cat in coords.keys():

        # fit line through a category of lines
        x = np.array([coord[0] for coord in coords[cat]])
        y = np.array([coord[1] for coord in coords[cat]])

        try:
            m, b = np.polyfit(x, y, 1)  # y = m * x + b

            # filter outliers
            x = filter_outliers(x)
            y = filter_outliers(y)

            # find line extreme coordinates
            min_y = int(img.shape[0] * 0.6)
            max_y = img.shape[0]
            min_x = int((min_y - b) / m)
            max_x = int((max_y - b) / m)

            cv2.line(img, (min_x, min_y), (max_x, max_y), color, thickness)
        except ValueError:
            pass


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for
    # processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)

    gray = grayscale(image)

    # Gaussian smoothing
    blur_gray = gaussian_blur(gray, 5)

    # Define our parameters for Canny and apply
    low_th = 50
    high_th = 150
    edges = canny(blur_gray, low_th, high_th)

    # Four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]),
                          (500, 310),
                          (imshape[1], imshape[0])]],
                        dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 4  # distance resolution of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 60     # minimum number of intersections in Hough grid cell
    min_line_length = 10  # minimum number of pixels making up a line
    max_line_gap = 150    # maximum gap between connectable line segments

    # Run Hough on edge detected image
    line_image = hough_lines(
        masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # Draw the lines on the edge image
    result = weighted_img(image, line_image)

    return result


img_path = os.path.expanduser('~/dev/learning/CarND-LaneLines-P1/test_images/')
img_files = os.listdir(img_path)
print('Test image path: {}'.format(img_path))
print('Test images: {}'.format(img_files))

for img in img_files:
    path = os.path.join(img_path, img)
    image = mpimg.imread(path)

    # printing out some stats and plotting
    print('This image is:', img, 'with dimesions:', image.shape)

    result = process_image(image)

    plt.imshow(result, cmap='gray')
    plt.savefig('lanes_{}'.format(img))
    plt.show()
