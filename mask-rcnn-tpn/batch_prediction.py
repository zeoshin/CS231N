import os, math
import json, copy
import sys
from os import listdir

image_list = listdir('/home/HSZHAO/final_project/pytorch-mask-rcnn/coco2017_data/val2017')
for i in range(len(image_list)):
        image_list[i] = image_list[i].split('.')[0]

# print(image_list)
batch_size = int(sys.argv[1])
print('batch size is set to {}'.format(batch_size))
total_image_num = len(image_list)

with open('coco2017_data/annotations/instances_val2017_orig.json') as json_data:
    d = json.load(json_data)
    # print("The keys are: {}".format(d.keys()))

    for i in range(0, total_image_num):
        d['images'][i]['file_name'] = image_list[i] + '.jpg'
        d['images'][i]['id'] = i
        d['annotations'][i]['image_id'] = i
    # start batch prediction
    start_from = 685
    for batch_id in range(math.ceil(total_image_num / batch_size)):
        if start_from > batch_id * batch_size and start_from > (batch_id + 1) * batch_size:
            continue
        
        #start_image_id = batch_id * batch_size
        if start_from > batch_id * batch_size and start_from < (batch_id + 1) * batch_size:
            start_image_id = start_from
        else:
            start_image_id = batch_id * batch_size
        end_image_id = min((batch_id + 1) * batch_size, total_image_num)
        print('creating json from image {} to {}'.format(start_image_id, end_image_id - 1))
        #with open('coco2017_data/annotations/instances_val2017.json', 'w') as f:
        with open('coco2017_data/annotations/part{}.json'.format(batch_id), 'w') as f:
            d_copy = copy.deepcopy(d)
            d_copy['images'] = d['images'][start_image_id:end_image_id]
            d_copy['annotations'] = d['annotations'][start_image_id:end_image_id]
            json.dump(d_copy, f, ensure_ascii=False)
        copy_json_command = 'cp ' +  'coco2017_data/annotations/part{}.json '.format(batch_id) + 'coco2017_data/annotations/instances_val2017.json'
        os.system(copy_json_command)
        run_evaluation_command = 'python coco.py evaluate --dataset=coco2017_data/ --year=2017 --model=mask_rcnn_coco.pth --limit={}'.format(batch_size) 
        print("Running batch predction from image {} to image {}".format(start_image_id, end_image_id - 1))
        os.system(run_evaluation_command)