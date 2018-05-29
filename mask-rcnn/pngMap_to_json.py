import json
import numpy as np
from pycocotools import mask
from skimage import measure
import cv2
import json, copy

from os import listdir
# pwd = '/home/HSZHAO/data/cvpr-2018-autonomous-driving'
# image_list = listdir(pwd + 'train_label')

# img = cv2.imread(filepath, -1)

mask_list = []
# for val in np.unique(img):
# 	if val != 255:
# 		mask_list.append(np.uint8(1) * (img == val))

ground_truth_binary_mask = np.array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                    [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                    [  0,   0,   0,   0,   0,   1,   1,   0,   0,   0],
                                    [  0,   0,   0,   0,   0,   1,   1,   0,   0,   0],
                                    [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                    [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                    [  1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                    [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                    [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=np.uint8)
# ground_truth_binary_mask = mask_list[0]
# print(ground_truth_binary_mask)
# print(type(ground_truth_binary_mask))
# print(ground_truth_binary_mask.dtype)
fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
ground_truth_area = mask.area(encoded_ground_truth)
ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
contours = measure.find_contours(ground_truth_binary_mask, 0.5)

info = {
        "url":"http://cocodataset.org",
        "date_created":"2017/09/01",
        "version":"1.0",
        "year":2017,
        "description":"COCO 2017 Dataset",
        "contributor":"COCO Consortium"   
}

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
    
print(json.dumps([info, annotation], indent=4))
with open('data.txt', 'w') as f:
  json.dump([info, annotation], f, ensure_ascii=False)


