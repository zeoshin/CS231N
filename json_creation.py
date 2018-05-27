import json, copy

from os import listdir
image_list = listdir('/Volumes/myDisk/kaggle/Segmentation/test')
for i in range(len(image_list)):
	image_list[i] = image_list[i].split('.')[0]

# print(image_list)

with open('instances_val2017_template.json') as json_data:
    d = json.load(json_data)
    print("The keys are: {}".format(d.keys()))
    d['images'][0]['file_name'] = image_list[0] + '.jpg'
    image_temp = copy.deepcopy(d['images'][0])
    annotation_temp = copy.deepcopy(d['annotations'][0])
    
    for i in range(1, len(image_list)):
        d['images'].append(copy.deepcopy(image_temp))
        d['annotations'].append(copy.deepcopy(annotation_temp))
        d['images'][i]['file_name'] = image_list[i] + '.jpg'
        d['images'][i]['id'] = i
        d['annotations'][i]['id'] = i
    with open('output_json.txt', 'w') as f:
        json.dump(d, f, ensure_ascii=False)

    # with open('output_json.txt') as json_data:
    # 	d = json.load(json_data)
    # 	print(len(d['images']))
