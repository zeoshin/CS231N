import json
import numpy as np
from pycocotools import mask
from skimage import measure
import cv2
import json, copy
import random
from os import listdir
import glob


def get_clip_images(file_name, cam5_list, cam6_list):
    clip_images = []
    if 'Camera_5' not in file_name:
        # cam6_list
        cam5_name = file_name.replace('Camera_6', 'Camera_5')
        if cam5_name not in cam5_list:
            return None

        idx = cam6_list.index(file_name)
        for i in range(idx - 4, idx + 1):
            clip_images.append(cam6_list[i])

        idx = cam5_list.index(cam5_name)
        for i in range(idx - 4, idx + 1):
            clip_images.append(cam5_list[i])
    else:
        cam6_name = file_name.replace('Camera_5', 'Camera_6')
        if cam6_name not in cam6_list:
            return None

        idx = cam5_list.index(file_name)
        for i in range(idx - 4, idx + 1):
            clip_images.append(cam5_list[i])

        idx = cam6_list.index(cam6_name)
        for i in range(idx - 4, idx + 1):
            clip_images.append(cam6_list[i])

    return clip_images


def create_json(output_file_name, ref_json, cam5_list, cam6_list):
    # read json template
    with open('instances_val2017_template.json') as json_data:
        d = json.load(json_data)
    with open(ref_json) as ref:
        d_ref = json.load(ref)

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
    missing_counter = 0
    for clip in d_ref['images']:
        clip_images = get_clip_images(clip['file_name'].split('.')[0] + '_instanceIds.png', cam5_list, cam6_list)
        if clip_images is None:
            missing_counter += 1
            continue
        for i, single_image in enumerate(clip_images):

            if counter % 50 == 0:
                print("{} images so far".format(counter))
            image_name = single_image
            file_name = image_name.replace( '_instanceIds', '')
            file_name = file_name.replace( '.png', '.jpg')
            json_single_image = copy.deepcopy(d['images'][0])
            json_single_image['file_name'] = file_name
            json_single_image['id'] = counter
        
            json_copy['images'].append(json_single_image)
            #print(json_copy['images'])
        
            filepath = pwd + '/train_label/' + image_name
            img = cv2.imread(filepath, -1)

            instance_label = np.unique(img)
            instance_label = instance_label[instance_label != 255].tolist()
            instance_label = [item for item in instance_label if item//1000 in kaggle_to_coco]
        
            id_list = list(map(lambda x: kaggle_to_coco[x//1000], instance_label))

            mask_list = []
            for val in instance_label:
                mask_list.append(np.uint8(1) * (img == val))
           #    print(mask_list)
            for j,mask_i in enumerate(mask_list):
                ground_truth_binary_mask = mask_i
                mask_sum = np.sum(mask_i)
                fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)

                encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
                ground_truth_area = mask.area(encoded_ground_truth)
                ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
                contours = measure.find_contours(ground_truth_binary_mask, 0.5)
                annotation = {
                    "segmentation": [],
                    "area": ground_truth_area.tolist(),
                    "iscrowd": 0,
                    "image_id": counter,
                    "bbox": ground_truth_bounding_box.tolist(),
                    "category_id": id_list[j],
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
            counter += 1
    with open(output_file_name + '.json', 'w') as f:
        json.dump(json_copy, f, ensure_ascii=False)
    print("missing counter is {}".format(missing_counter))


np.random.seed(1234)
pwd = '/home/HSZHAO/data/cvpr-2018-autonomous-driving'
ref_json = 'val_kaggle.json'
cam5_list = sorted(glob.glob(pwd + '/train_label/*Camera_5*'))
cam5_list = [img.split('/')[-1] for img in cam5_list]
cam6_list = sorted(glob.glob(pwd + '/train_label/*Camera_6*'))
cam6_list = [img.split('/')[-1] for img in cam6_list]
print(len(cam5_list))
print(len(cam6_list))
create_json('instance_train_900_clips_dual_cam', ref_json, cam5_list, cam6_list)
