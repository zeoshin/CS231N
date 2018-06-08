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

def create_json(output_file_name, isTrain, deduplicate_set, img_set, num_of_clip):
    #pwd = '/home/HSZHAO/data/cvpr-2018-autonomous-driving'
    #image_list = listdir(pwd + '/train_label')

    # read json template

    start_img_set = set()

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

    pwd = '/home/HSZHAO/data/cvpr-2018-autonomous-driving'
    txt_list = listdir(pwd + '/train_video_list')

    annotation_counter = 0

    for counter in range(num_of_clip):
        if counter % 50 == 0:
            print("{} clips so far".format(counter))
        
        txt_idx = random.randint(0, len(txt_list) - 1)
        list_file = open(pwd + '/train_video_list/' + txt_list[txt_idx])
        image_list = list_file.readlines()
        image_list = [img.split('\\')[-1].strip() for img in image_list]
        image_idx = random.randint(0, len(image_list) - 5)

        valid_count = 0
        while (image_list[image_idx] in start_img_set) or valid_count != 5:
            valid_count = 0
            txt_idx = random.randint(0, len(txt_list) - 1)
            list_file = open(pwd + '/train_video_list/' + txt_list[txt_idx])
            image_list = list_file.readlines()
            image_list = [img.split('\\')[-1].strip() for img in image_list]
            image_idx = random.randint(0, len(image_list) - 5)
            for k in range(image_idx, image_idx + 5):
                image_name = image_list[k]                
                filepath = pwd + '/train_label/' + image_name
                if not os.path.isfile(filepath) or (not isTrain and image_name in deduplicate_set):
                    print(image_name)
                    break
                valid_count += 1
        start_img_set.add(image_list[image_idx])
        for i in range(image_idx, image_idx + 5):
            image_name = image_list[i]
            file_name = image_name.replace( '_instanceIds', '')
            file_name = file_name.replace( '.png', '.jpg')
            json_single_image = copy.deepcopy(d['images'][0])
            json_single_image['file_name'] = file_name
            json_single_image['id'] = counter * 5 + i - image_idx 
            print("Current img_id is {}".format(json_single_image['id'])) 
            json_copy['images'].append(json_single_image)
            #print(json_copy['images'])
            
            if True:
                filepath = pwd + '/train_label/' + image_name
                img = cv2.imread(filepath, -1)
                
#                print(image_name)
                instance_label = np.unique(img)
#                print(instance_label)
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
                        "image_id": counter * 5 + i  - image_idx,
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
            img_set.add(image_name)
            #print(json.dumps(annotation, indent=4))
            #print(mask_sum)
        #exit(0)
    with open(output_file_name + '.json', 'w') as f:
        json.dump(json_copy, f, ensure_ascii=False)


np.random.seed(1234)
img_set_train = set()
create_json('single_cam_multi_clips_kaggle_train', True, None, img_set_train, 900)
img_set_val = set()
create_json('single_cam_multi_clips_kaggle_val', False, img_set_train, img_set_val, 350)
