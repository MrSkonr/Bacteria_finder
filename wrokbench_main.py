import numpy as np
import matplotlib.pyplot as plt
import cv2

bacteria = cv2.imread('Staph. aureus, 2-2.bmp')


def select_colorsp(img, colorsp='gray'):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Split BGR
    red, green, blue = cv2.split(img)
    # Convert to HSV
    im_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split HSV
    hue, sat, val = cv2.split(im_hsv)
    # Store channels in a dict
    channels = {'gray': gray, 'red': red, 'green': green,
                'blue': blue, 'hue': hue, 'sat': sat, 'val': val}

    return channels[colorsp]

def display(im_left, im_right, name_l='Left', name_r='Right', figsize=(12,10)):

    # Flip channels for display if RGB as matplotlib requires RGB
    im_l_dis = im_left[...,::-1] if len(im_left.shape) > 2 else im_left
    im_r_dis = im_right[...,::-1] if len(im_right.shape) > 2 else im_right

    plt.figure(figsize=figsize)

    plt.subplot(121)
    plt.imshow(im_l_dis)
    plt.title(name_l)
    plt.axis(False)

    plt.subplot(122)
    plt.imshow(im_r_dis)
    plt.title(name_r)
    plt.axis(False)

    plt.show()

def threshold(img, thresh=127, mode='inverse'):
    im = img.copy()

    if mode == 'direct':
        thresh_mode = cv2.THRESH_BINARY
    else:
        thresh_mode = cv2.THRESH_BINARY_INV

    ret, thresh = cv2.threshold(im, thresh, 255, thresh_mode)

    return thresh

# Select colorspace
gray_bacteria = select_colorsp(bacteria)
# Perform thresholding
thresh_bacteria = threshold(gray_bacteria, thresh=180, mode='direct')

cv2.imwrite("test.jpg", thresh_bacteria)

print('Done')
