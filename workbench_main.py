import numpy as np
import matplotlib.pyplot as plt
import cv2

bacteria = cv2.imread('test_input.bmp')


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

def get_boxes(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort according to the area of contours in descending order
    sorted_cnt = sorted(contours, key = cv2.contourArea, reverse= True)
    # Remove max area, outermost contour
    sorted_cnt.remove(sorted_cnt[0])
    bboxes = []
    for cnt in sorted_cnt:
        x, y, w, h = cv2.boundingRect(cnt)
        cnt_area = w * h
        bboxes.append((x, y, x+w, y+h))
    return bboxes

def draw_annotations(img, bboxes, thickness = 2, color = (0, 255, 0)):
    annotations = img.copy()
    for box in bboxes:
        tlc = (box[0], box[1])
        brc = (box[2], box[3])
        cv2.rectangle(annotations, tlc, brc, color, thickness, cv2.LINE_AA)
    
    return annotations

def morph_op(img, mode = 'open', ksize = 5, iterations = 1):
    im = img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    if mode == 'open':
        morphed = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    elif mode == 'close':
        morphed = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    elif mode == 'erode':
        morphed = cv2.erode(im, kernel)
    else:
        morphed = cv2.dilate(im, kernel)

    for iter in range(iterations - 1):
        im = morphed.copy()
        if mode == 'open':
            morphed = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
        elif mode == 'close':
            morphed = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
        elif mode == 'erode':
            morphed = cv2.erode(im, kernel)
        else:
            morphed = cv2.dilate(im, kernel)

    return morphed

def get_filtered_bboxes(img, min_area_ratio = 0.001):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort according to the area of contours in descending order
    sorted_cnt = sorted(contours, key = cv2.contourArea, reverse= True)
    # Remove max area, outermost contour
    sorted_cnt.remove(sorted_cnt[0])
    bboxes = []
    # Image area
    im_area = img.shape[0] * img.shape[1]
    for cnt in sorted_cnt:
        x, y, w, h = cv2.boundingRect(cnt)
        cnt_area = w * h
        # Remove very small detections
        if cnt_area > min_area_ratio * im_area:
            bboxes.append((x, y, x+w, y+h))
    return bboxes

# Select colorspace
gray_bacteria = select_colorsp(bacteria)
# Perform thresholding
thresh_bacteria = threshold(gray_bacteria, thresh=160, mode='direct')
# Save the thresholded image
cv2.imwrite("test_output_thresholded.jpg", thresh_bacteria)

# Switch back to RGB image
thresh_bacteria_rgb = cv2.cvtColor(thresh_bacteria, cv2.COLOR_GRAY2RGB)
# Draw bounding boxes
bounded_bacteria = draw_annotations(thresh_bacteria_rgb, get_boxes(thresh_bacteria), thickness= 1, color= (255, 0, 0))
# Save the bounded image
cv2.imwrite("test_output_bounded.jpg", bounded_bacteria)
# Draw bounding boxes in original image
bounded_bacteria_original = draw_annotations(bacteria, get_boxes(thresh_bacteria), thickness= 1, color= (255, 0, 0))
# Save the bounded image
cv2.imwrite("test_output_bounded_original.jpg", bounded_bacteria_original)

# Perform morphological operation
morphed_bacteria = morph_op(thresh_bacteria, mode = 'close', iterations= 1)
# Save the morphed image
cv2.imwrite("test_output_morphed.jpg", morphed_bacteria)

print('Done')
