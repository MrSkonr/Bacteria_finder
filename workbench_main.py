from Annotation_functions import *

# bacteria = cv2.imread('test_input_small.jpg')
bacteria = cv2.imdecode(np.fromfile('test_input_small.jpg', dtype=np.uint8), cv2.IMREAD_UNCHANGED)

print(bacteria.shape)

# Select colorspace
gray_bacteria = select_colorsp(bacteria)
# Perform thresholding
thresh_bacteria = threshold(gray_bacteria, thresh=160, mode='direct')
# Save the thresholded image
cv2.imwrite("test_output_small_thresholded.jpg", thresh_bacteria)

# # Switch back to RGB image
# thresh_bacteria_rgb = cv2.cvtColor(thresh_bacteria, cv2.COLOR_GRAY2RGB)
# # Draw bounding boxes
# bounded_bacteria = draw_annotations(thresh_bacteria_rgb, get_boxes(thresh_bacteria), thickness= 1, color= (255, 0, 0))
# # Save the bounded image
# cv2.imwrite("test_output_bounded.jpg", bounded_bacteria)
# # Draw bounding boxes in original image
# bounded_bacteria_original = draw_annotations(bacteria, get_boxes(thresh_bacteria), thickness= 1, color= (255, 0, 0))
# # Save the bounded image
# cv2.imwrite("test_output_bounded_original.jpg", bounded_bacteria_original)

# # Perform morphological operation
# morphed_bacteria = morph_op(thresh_bacteria, mode = 'close', iterations= 1)
# # Save the morphed image
# cv2.imwrite("test_output_morphed.jpg", morphed_bacteria)

# # Testing saving annotations
# save_annotations(bacteria, get_boxes(thresh_bacteria))

print('Done')
