import numpy as np
import matplotlib.pyplot as plt
import cv2

def select_colorsp(img, colorsp='gray'):
    '''
    Function, that transforms the original image into one of color spaces (gray)
    
    input:
    img --- image in np.array HxWx3 format
    colorsp --- the desired color space

    output:
    channels[colorsp] --- np.array HxWx3 in specified color space
    '''

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
    '''
    Function, that displays two specified images

    input: 
    im_left: np.array HxWxn image
    im_right: np.array HxWxn image
    name_l: string, wil be displayed as title of the left image
    name_r: string, wil be displayed as title of the right image
    figsize: tuple with size of 2. Size of the figure

    output: matplotlib instance with two images shown
    '''

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
    '''
    Perform the basic binarization by stating the threshold value

    input:
    img: np.array HxW gray image
    thresh: int, desired threshold
    mode: the desired binary state of final image. 'direct' or 'inverse'

    output:
    thresh: np.array HxW binarized image
    '''

    im = img.copy()

    if mode == 'direct':
        thresh_mode = cv2.THRESH_BINARY
    else:
        thresh_mode = cv2.THRESH_BINARY_INV

    ret, thresh = cv2.threshold(im, thresh, 255, thresh_mode)

    return thresh

def get_boxes(img):
    '''
    Basic boxes finder of binarized image

    input:
    img: np.array HxW binarized image

    output:
    bboxes: np.array Nx4 with boxes coordinates
    '''

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
    '''
    Function to draw boxes in specified coordinates

    input:
    img: np.array HxWxn image
    bboxes: np.array Nx4 with boxes coordinates
    thickness: int, the thickness of the drawn line
    color: int tuple with the size of 3. color in BGR format

    output:
    annotations: np.array HxWxn image with drawn boxes
    '''
    annotations = img.copy()
    for box in bboxes:
        tlc = (box[0], box[1])
        brc = (box[2], box[3])
        cv2.rectangle(annotations, tlc, brc, color, thickness, cv2.LINE_AA)
    
    return annotations

def morph_op(img, mode = 'open', ksize = 5, iterations = 1):
    '''
    Function to perform basic morphological operations

    input:
    img: np.array HxWxn image
    mode: string, desired operation. Possible variants are 'dilate', 'erode', 'close', 'open'
    ksize: int, size of the square kernel
    iterations: number of times the operation will take place

    output:
    morphed: changed np.array HxWxn image
    '''
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
    '''
    Basic boxes finder of binarized image with simple size filter

    input:
    img: np.array HxW binarized image
    min_area_ratio: int, ratio, that determines the sizes of removed boxes 
    in the comparison to original image

    output:
    bboxes: np.array Nx4 with boxes coordinates
    '''
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

def get_color_mask(img, lower = [0,0,0], upper = [0,255,255]):
    '''
    Function to mask-out the image in specified HSV boundaries

    input:
    img: np.array HxWxn image
    lower: np.array with the size of 3. Lower HSV boundaries
    upper: np.array with the size of 3. Lower HSV boundaries

    output:
    inv_mask: np.array HxWxn masked-out image
    '''
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low = np.array(lower)
    up = np.array(upper)
    mask = cv2.inRange(img_hsv, low, up)
    inv_mask = 255 - mask

    return inv_mask

def save_annotations(img, bboxes):
    '''
    Function to save boxes locations in txt file in YOLO format

    input:
    img: np.array HxWxn image
    bboxes: np.array Nx4 with boxes coordinates

    output:
    Txt file 'bounding_boxes_yolo.txt' will be generated in the origin folder
    '''
    img_height = img.shape[0]
    img_width = img.shape[1]
    with open('bounding_boxes_yolo.txt', 'w') as f:
        for box in bboxes:
            x1, y1 = box[0], box[1]
            x2, y2 = box[2], box[3]

            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            width = x2 - x1
            height = y2 - y1
            x_centre, y_centre = int(width/2), int(height/2)

            norm_xc = x_centre/img_width
            norm_yc = y_centre/img_height
            norm_width = width/img_width
            norm_height = height/img_height

            yolo_annotations = ['0', ' ' + str(norm_xc),
                                ' ' + str(norm_yc),
                                ' ' + str(norm_width),
                                ' ' + str(norm_height), '\n']
            
            f.writelines(yolo_annotations)