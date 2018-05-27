import json
import numpy as np
from pycocotools import mask
from skimage import measure
import cv2
import json, copy

from os import listdir
pwd = '/home/HSZHAO/data/cvpr-2018-autonomous-driving'
image_list = listdir(pwd + '/train_label')

for image in image_list:
    filepath = pwd + '/train_label/' + '171206_034513181_Camera_6_instanceIds.png'
    img = cv2.imread(filepath, -1)

    mask_list = []
    for val in np.unique(img):
        if val != 255:
            mask_list.append(np.uint8(1) * (img == val))
    for mask_i in mask_list:
        mask_sum = np.sum(mask_i)
        ground_truth_binary_mask = mask_i 
# print(ground_truth_binary_mask)
# print(type(ground_truth_binary_mask))
# print(ground_truth_binary_mask.dtype)
        fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)

        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        ground_truth_area = mask.area(encoded_ground_truth)
        ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
        contours = measure.find_contours(ground_truth_binary_mask, 0.5)
        annotation = {
            "segmentation": [],
            "area": ground_truth_area.tolist(),
            "iscrowd": 0,
            "image_id": 123,
            "bbox": ground_truth_bounding_box.tolist(),
            "category_id": 1,
            "id": 1
        }
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            annotation["segmentation"].append(segmentation)
    
        print(json.dumps(annotation, indent=4))
        print(mask_sum)
    exit(0)


