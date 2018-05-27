import json
import numpy as np
from pycocotools import mask
from skimage import measure
import cv2
import json, copy

from os import listdir
pwd = '/home/HSZHAO/data/cvpr-2018-autonomous-driving'
image_list = listdir(pwd + '/train_label')


kaggle_to_coco = {
    36: 1, #person
    35: 2, #bicycle
    33: 3, #car
    34: 4, #motorcycle
    39: 6, #bus
    38: 8, #truck
    40: 2 #tricycle => bicycle
}

for image in image_list:
    filepath = pwd + '/train_label/' + '171206_034513181_Camera_6_instanceIds.png'
    img = cv2.imread(filepath, -1)
    instance_label = np.unique(img)
    instance_label = instance_label[instance_label != 255]

    id_list = list(map(lambda x: kaggle_to_coco[x//1000], instance_label))

    mask_list = []
    for val in instance_label:
        mask_list.append(np.uint8(1) * (img == val))
    for i,mask_i in enumerate(mask_list):
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
            "category_id": id_list[i],
            "id": i
        }
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            annotation["segmentation"].append(segmentation)
    
        print(json.dumps(annotation, indent=4))
        print(mask_sum)
    exit(0)


