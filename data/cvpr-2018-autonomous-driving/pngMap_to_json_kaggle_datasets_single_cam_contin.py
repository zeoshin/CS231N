import json
import numpy as np
from pycocotools import mask
from skimage import measure
import cv2
import json, copy
import random
from os import listdir
import os
import glob
def create_json(sample_idx, output_file_name, image_list):
    #pwd = '/home/HSZHAO/data/cvpr-2018-autonomous-driving'
    #image_list = listdir(pwd + '/train_label')

    # read json template
    with open('instances_val2017_template.json') as json_data:
        d = json.load(json_data)
    json_copy = copy.deepcopy(d)
    json_copy['images'] = []
    json_copy['annotations'] = []

    kaggle_to_coco = {
        36: 1, #person
        35: 2, #bicycle
        33: 3, #car
        34: 4, #motorcycle
        39: 6, #bus
        38: 8, #truck
        40: 2 #tricycle => bicycle
    }

    annotation_counter = 0
    counter = 0
    for i in sample_idx:
        if counter % 50 == 0:
            print("{} images so far".format(counter))
        counter += 1
        #if i % 50 == 0:
        #    print("Current image is {}".format(i))
        image_name = image_list[i]
        file_name = image_name.replace( '_instanceIds', '')
        file_name = file_name.replace( '.png', '.jpg')
        json_single_image = copy.deepcopy(d['images'][0])
        json_single_image['file_name'] = file_name
        json_single_image['id'] = i
        
        json_copy['images'].append(json_single_image)
        #print(json_copy['images'])
        
        filepath = pwd + '/train_label/' + image_name
        img = cv2.imread(filepath, -1)
        instance_label = np.unique(img)
        instance_label = instance_label[instance_label != 255].tolist()
        instance_label = [item for item in instance_label if item//1000 in kaggle_to_coco]
    #    print(instance_label)
        
        id_list = list(map(lambda x: kaggle_to_coco[x//1000], instance_label))

        mask_list = []
        for val in instance_label:
            mask_list.append(np.uint8(1) * (img == val))
    #    print(mask_list)
        for j,mask_i in enumerate(mask_list):
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
                "category_id": id_list[j],
                "id": annotation_counter
            }
            annotation_counter += 1
            for contour in contours:
                contour = np.flip(contour, axis=1)
                segmentation = contour.ravel().tolist()
                annotation["segmentation"].append(segmentation)
                # print(segmentation)
            json_copy['annotations'].append(annotation)

            #print(json.dumps(annotation, indent=4))
            #print(mask_sum)
        #exit(0)
    with open(output_file_name + '.json', 'w') as f:
        json.dump(json_copy, f, ensure_ascii=False)


np.random.seed(1234)
pwd = '/home/HSZHAO/data/cvpr-2018-autonomous-driving'
list_file = open(pwd + '/train_video_list/road01_cam_5_video_6_image_list_train.txt')
image_list = list_file.readlines()
image_list = [img.split('\\')[-1].strip() for img in image_list]
continuous_idx = list(range(len(image_list)))
create_json(continuous_idx, 'single_cam_continuous_kaggle', image_list)


