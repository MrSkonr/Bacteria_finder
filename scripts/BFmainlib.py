import os

from numpy import argmax, histogram, zeros, mean, array, uint8, float16
from cv2 import medianBlur, drawContours, contourArea, findContours, threshold, cvtColor, imread, resize, \
                INTER_CUBIC, INTER_LANCZOS4, COLOR_RGB2HSV, THRESH_BINARY, THRESH_BINARY_INV, THRESH_TOZERO, THRESH_TOZERO_INV, RETR_LIST, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE
from skimage.measure import block_reduce

def HSV_transform(image : array) -> array:
    '''
    Transform RGB image to HSV color space
    '''
    return cvtColor(image, COLOR_RGB2HSV)
    
def HSV_threshold(image : array, type : str = 'simple') -> array:
    '''
    Perform thresholding of HSV image (hue channel)

    'simple' : threshold value is 60 (green color) and strategy is THRESH_BINARY

    'bact' : some area (not one value) in hue space is set to zero (corresponding to background).
    The 'bact' strategy seraches for maximum value in histogram and sets to zero every color +- margin (hardcoded) from argmax

    'color': some area (hardcoded) in hue space is set to its original value, the rest is set to zero.
    This strategy puts to zero every "non-red" and "non-purple" value
    '''
    # TODO: remove hardcode
    assert type == 'simple' or type == 'bact' or type == 'color'

    if type == 'simple':
        return threshold(image[:,:,0], 60, 180, THRESH_BINARY)[1]
    elif type == 'bact':
        margin = 5
        image_hue_hist = histogram(image[:,:,0].flatten(), bins=[el for el in range(181)])
        background_max = argmax(image_hue_hist[0])

        return threshold(image[:,:,0], background_max + margin, 181, THRESH_BINARY)[1] + threshold(image[:,:,0], background_max - margin, 181, THRESH_BINARY_INV)[1]
    elif type == 'color':
        image_hue_hist = histogram(image[:,:,0].flatten(), bins=[el for el in range(181)])
        margin_min = argmax(image_hue_hist[0])-5
        margin_max = argmax(image_hue_hist[0])+30

        return threshold(image[:,:,0], margin_min, 181, THRESH_BINARY_INV)[1] + threshold(image[:,:,0], margin_max, 181, THRESH_BINARY)[1]
    
def HSV_segmenting(image : array) -> list:
    '''
    Segment an image by thresholding it in HSV colorspace
    '''
    # TODO: add different types of HSV segmenting
    image_hsv = HSV_transform(image)
    image_hsv_mask = HSV_threshold(image_hsv, type = 'bact')
    contours_hsv, hierarchy_hsv = findContours(image_hsv_mask, RETR_LIST, CHAIN_APPROX_SIMPLE)
    contours_hsv_mask = sorted(contours_hsv, key = contourArea, reverse= True)

    return contours_hsv_mask

class BF_image():
    def __init__(self, verbose = False):
        '''
        BF_image: base class of Bacteria Finder program

        Class is initializing with dark image and zero objects
        '''
        self.bacteria_image_loaded = zeros((1216, 1616, 3))
        self.bacteria_image_preprocessed = zeros((1216, 1616, 3))
        self.objects_db = {}

        self.verbose = verbose

    def object_new_add(self, contour : array):
        '''
        Function, that adds an object to BF_image class
        '''
        new_id = str(len(self.objects_db.keys()) + 1)
        self.objects_db[new_id] = BF_object(new_id, 'undefined', contour)

        if self.verbose:
            print(f'Added object {new_id=}')
    
    def load_image(self, path : str):
        '''
        Function, that loads image and replace channels from BRG to RGB
        '''
        # TODO: add possible fixes for utf names
        assert os.path.isfile(path)
        self.bacteria_image_loaded = imread(path)
        self.bacteria_image_loaded = self.bacteria_image_loaded[:,:,::-1]

        if self.verbose:
            print('Successfully loaded an image')

    def preprocess_image(self):
        '''
        Function, that performs preprocessing of a loaded image

        As for now image preprocessing consists of shrinking by average pooling and restoring shape by
        cubic interpolation
        '''
        # TODO: add different methods and swithes in the future
        self.bacteria_image_preprocessed = resize(uint8(
            block_reduce(self.bacteria_image_loaded, (2,2,1), mean, func_kwargs={'dtype': float16})),
            self.bacteria_image_loaded.shape[1::-1], interpolation = INTER_CUBIC)
        
        self.bacteria_image_preprocessed = medianBlur(self.bacteria_image_preprocessed, 5)
        
        if self.verbose:
            print('Successfully preprocessed an image')
        
    def segment_image(self):
        '''
        Main pipeline for segmentation
        '''
        contours_hsv_mask = HSV_segmenting(self.bacteria_image_preprocessed)
        for contour in contours_hsv_mask:
            if contourArea(contour) > 50:
                self.object_new_add(contour)

        if self.verbose:
            print('Successfully HSV:segmented an image')

    def segment_draw(self) -> array:
        '''
        Draw segmented image depending on the current state of segmentation
        '''
        objects_undefined_list = []
        for object in self.objects_db.values():
            if object.object_type == 'undefined':
                objects_undefined_list.append(object.object_countour_coords)

        return drawContours(self.bacteria_image_preprocessed.copy(),
                            objects_undefined_list,
                            contourIdx=-1, color=(0, 255, 0), thickness=1)

class BF_object():
    def __init__(self, id : str, type : str, contour_coords : list):
        '''
        BF_object: object on an image class of Bacteria Finder program

        type: undefined, bacilli, coccus, group, misc
        '''
        self.object_id = id
        self.object_type = type
        self.object_countour_coords = contour_coords