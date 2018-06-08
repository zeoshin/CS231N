import json, copy

from os import listdir
image_list = listdir('/home/HSZHAO/final_project/pytorch-mask-rcnn/coco2017_data/val2017')
for i in range(len(image_list)):
	image_list[i] = image_list[i].split('.')[0]

# print(image_list)

with open('coco2017_data/annotations/instances_val2017_orig.json') as json_data:
    d = json.load(json_data)
    print("The keys are: {}".format(d.keys()))
    
    for i in range(0, len(image_list)):
        d['images'][i]['file_name'] = image_list[i] + '.jpg'
        d['images'][i]['id'] = i
        d['annotations'][i]['image_id'] = i
    with open('coco2017_data/annotations/instances_val2017.json', 'w') as f:
        d['images'] = d['images'][:1917]
        d['annotations'] = d['annotations'][:1917]
        json.dump(d, f, ensure_ascii=False)

    # with open('output_json.txt') as json_data:
    # 	d = json.load(json_data)
    # 	print(len(d['images']))
