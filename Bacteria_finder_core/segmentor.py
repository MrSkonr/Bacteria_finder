
from sys import path

path.append('../Bacteria_finder')
path.append('../Bacteria_finder/Bacteria_finder_core')
from cellpose import core
from cellpose_omni import models
from classifier import MobileNetV2
from cv2 import (COLOR_BGR2RGB, THRESH_TRIANGLE, THRESH_TRUNC, bilateralFilter,
                 cvtColor, threshold)
from numpy import (append, float32, max, min, pad, transpose, uint8, unique,
                   where, zeros_like)
from omnipose.utils import normalize99
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, mark_boundaries, watershed
from torch import cuda, device, load, tensor


class Bacteria_segmentor():

    def __init__(self, segmentor_name = "watershed") -> None:
        self.segmentor_name = segmentor_name
        self.model = None

        self.device = 'cuda' if cuda.is_available() else 'cpu'
        # device = 'cpu'
        self.classifier_model = MobileNetV2().to(self.device)
        
        self.classifier_model.load_state_dict(load('Bacteria_finder_core/trained_model_masks_MobileNetV2_2M_v3.pt',
                                                          map_location=device(self.device)))
        self.classifier_model.eval()

    def Grayscaling(self, image):
        # return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:,:,0]
        # return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)[:,:,0]
        return image[:,:,1]
    
    def Filtering(self, image):
        #cv2.GaussianBlur(one_channel_bacteria, (5,5), 0)
        return bilateralFilter(image, d=5, sigmaColor=40, sigmaSpace=40)
    
    def Thresholding(self, image):
        return threshold(image,0,255,THRESH_TRUNC+THRESH_TRIANGLE)[1]
    
    def Segmentation(self, image):
        if self.segmentor_name == "watershed":
            return watershed(sobel(image))
        elif self.segmentor_name == "felzenszwalb":
            return felzenszwalb(image)
        elif self.segmentor_name == "omnipose":
            use_GPU = core.use_gpu()
            model_name = 'bact_phase_omni'
            # model_name = "bact_phase_omnitorch_0"
            self.model = models.CellposeModel(gpu=use_GPU, model_type=model_name)
            chans = [0,0]
            # define parameters
            mask_threshold = -1 
            verbose = 0
            transparency = True
            rescale=None
            omni = True
            flow_threshold = 0
            resample = True
            cluster=True
            return self.model.eval(normalize99(image),channels=chans,rescale=rescale,mask_threshold=mask_threshold,
                                    transparency=transparency,flow_threshold=flow_threshold,omni=omni,
                                    cluster=cluster, resample=resample,verbose=verbose)[0]
        
    def get_bboxs(self, mask_img, mask_list):
        # function to return the list of coordinates of boxes
        result = []
        for mask_num in mask_list:
            coords = where(mask_img == mask_num)
            bbox = [min(coords[0]), max(coords[0]), min(coords[1]), max(coords[1])]
            result.append(bbox)
        return result
    
    def Classify(self, bboxes):
        for i, bbox in enumerate(bboxes):
            ROI_img = self.bacteria[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1]
            ROI_img_classify = ROI_img
            ROI_mask = self.one_channel_bacteria_filtered_thresholded_labels[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1]
            ROI_mask = where(ROI_mask == i + 1, ROI_mask, 0).copy()
            ROI_mask = where(ROI_mask == 0, ROI_mask, 1).copy()

            if ROI_img.shape[0] > 172 or ROI_img.shape[1] > 172:
                self.one_channel_bacteria_misc_labels[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1] = ROI_mask*(i+1)
                continue

            label = self.classify_bboxs(ROI_img_classify, ROI_mask)
            if label[0] == 0:
                self.one_channel_bacteria_bacili_labels[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1] = ROI_mask*(i+1)
            elif label[0] == 1:
                self.one_channel_bacteria_cocci_labels[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1] = ROI_mask*(i+1)
            elif label[0] == 2:
                self.one_channel_bacteria_grouped_labels[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1] = ROI_mask*(i+1)
            elif label[0] == 3:
                self.one_channel_bacteria_misc_labels[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1] = ROI_mask*(i+1)

    def classify_bboxs(self, img, mask):
        padded_image = cvtColor(self.image_padder(img), COLOR_BGR2RGB)
        padded_image = transpose(padded_image, axes=[2, 0, 1])
        padded_mask = self.mask_padder(mask)
        padded_mask = padded_mask.reshape(1, padded_mask.shape[0], padded_mask.shape[1])
        input_tensor = tensor(append(padded_image.astype(float32)/255, padded_mask, axis = 0)).to(self.device)
        label = self.classifier_model(input_tensor.reshape(1, input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2])).argmax(1).detach().cpu().numpy()
        return label
    
    def image_padder(self, img):
        current_image = img
        pad_size = 172

        if current_image.shape[0] % 2 != 0:
            axis_0_pad_0 = int((pad_size - current_image.shape[0])/2) + 1
        else:
            axis_0_pad_0 = int((pad_size - current_image.shape[0])/2)
        axis_0_pad_1 = int((pad_size - current_image.shape[0])/2)

        if current_image.shape[1] % 2 != 0:
            axis_1_pad_0 = int((pad_size - current_image.shape[1])/2) + 1
        else:
            axis_1_pad_0 = int((pad_size - current_image.shape[1])/2)
        axis_1_pad_1 = int((pad_size - current_image.shape[1])/2)

        current_image = pad(current_image, ((axis_0_pad_0, axis_0_pad_1),
                                        (axis_1_pad_0, axis_1_pad_1),(0, 0)),
                                        mode='constant', constant_values=0)
        
        return current_image

    def mask_padder(self, mask):
        current_mask = mask
        pad_size = 172

        if current_mask.shape[0] % 2 != 0:
            axis_0_pad_0 = int((pad_size - current_mask.shape[0])/2) + 1
        else:
            axis_0_pad_0 = int((pad_size - current_mask.shape[0])/2)
        axis_0_pad_1 = int((pad_size - current_mask.shape[0])/2)

        if current_mask.shape[1] % 2 != 0:
            axis_1_pad_0 = int((pad_size - current_mask.shape[1])/2) + 1
        else:
            axis_1_pad_0 = int((pad_size - current_mask.shape[1])/2)
        axis_1_pad_1 = int((pad_size - current_mask.shape[1])/2)

        current_mask = pad(current_mask, ((axis_0_pad_0, axis_0_pad_1), (axis_1_pad_0, axis_1_pad_1)),mode='constant', constant_values=0)

        return current_mask


    def pipeline(self, in_image, mode = "all"):

        self.bacteria = in_image
        self.one_channel_bacteria = self.Grayscaling(self.bacteria)
        self.one_channel_bacteria_filtered = self.Filtering(self.one_channel_bacteria)
        self.one_channel_bacteria_filtered_thresholded = self.Thresholding(self.one_channel_bacteria_filtered)
        self.one_channel_bacteria_filtered_thresholded_labels = self.Segmentation(self.one_channel_bacteria_filtered_thresholded)

        if mode != "all":
            self.image_out = uint8(mark_boundaries(cvtColor(self.bacteria, COLOR_BGR2RGB), 
                                        self.one_channel_bacteria_filtered_thresholded_labels, 
                                        color=(28/255, 37/255, 22/255))*255)[:,:,::-1]
            return self.image_out
        else:
            self.one_channel_bacteria_bacili_labels = zeros_like(self.one_channel_bacteria_filtered_thresholded_labels)
            self.one_channel_bacteria_cocci_labels = zeros_like(self.one_channel_bacteria_filtered_thresholded_labels)
            self.one_channel_bacteria_grouped_labels = zeros_like(self.one_channel_bacteria_filtered_thresholded_labels)
            self.one_channel_bacteria_misc_labels = zeros_like(self.one_channel_bacteria_filtered_thresholded_labels)
            # obtaining bounding boxes and classifying them
            self.bboxes = self.get_bboxs(self.one_channel_bacteria_filtered_thresholded_labels, unique(self.one_channel_bacteria_filtered_thresholded_labels)[1:])
            self.image_out = cvtColor(self.bacteria, COLOR_BGR2RGB)
            self.Classify(self.bboxes)
            self.image_out = mark_boundaries(self.image_out, self.one_channel_bacteria_bacili_labels, color=(0, 128/255, 0)) 
            self.image_out = mark_boundaries(self.image_out, self.one_channel_bacteria_cocci_labels, color=(66/255, 170/255, 255/255))
            self.image_out = mark_boundaries(self.image_out, self.one_channel_bacteria_grouped_labels, color=(139/255, 0, 1))
            self.image_out = mark_boundaries(self.image_out, self.one_channel_bacteria_misc_labels, color=(0, 0, 0))
            return uint8(self.image_out*255)[:,:,::-1]








