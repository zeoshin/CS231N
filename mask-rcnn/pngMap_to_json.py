import json
import numpy as np
from pycocotools import mask
from skimage import measure
import cv2
import json, copy

from os import listdir
pwd = '/home/HSZHAO/data/cvpr-2018-autonomous-driving'
image_list = listdir(pwd + '/train_label')

# read json template
with open('instances_val2017_template.json') as json_data:
    d = json.load(json_data)
json_copy = copy.deepcopy(d)
json_copy['images'] = []
json_copy['annotations'] = []

annotation_counter = 0

for i in range(len(image_list)):
    image_name = image_list[i]
    file_name = image_name.replace( '_instanceIds', '')
    json_single_image = copy.deepcopy(d['images'][0])
    json_single_image['file_name'] = file_name
    json_single_image['id'] = i
    
    json_copy['images'].append(json_single_image)
    #print(json_copy['images'])
    
    filepath = pwd + '/train_label/' + image_name
    img = cv2.imread(filepath, -1)

    mask_list = []
    for val in np.unique(img):
        if val != 255:
            mask_list.append(np.uint8(1) * (img == val))
    for mask_i in mask_list:
        ground_truth_binary_mask = mask_i
        mask_sum = np.sum(mask_i)
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
            "image_id": i,
            "bbox": ground_truth_bounding_box.tolist(),
            "category_id": 1,
            "id": annotation_counter
        }
        annotation_counter += 1
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            annotation["segmentation"].append(segmentation)

        json_copy['annotations'].append(annotation)

        #print(json.dumps(annotation, indent=4))
        #print(mask_sum)
    #exit(0)
with open('output.json', 'w') as f:
    json.dump(json_copy, f, ensure_ascii=False)
