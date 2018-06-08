import numpy as np
import json
import pickle
import csv
import re
import datetime

# kaggle_categories = {
# 	"car": 33,
# 	"motorcycle": 34, # changed from motorbicycle
# 	"bicycle": 35,
# 	"person": 36,
# 	"rider": 37,
# 	"truck": 38,
# 	"bus": 39,
# 	"tricycle": 40, # not contained in coco
# 	"others": 0,
# 	"rover": 1,
# 	"sky": 17,
# 	"car_groups": 161,
# 	"motorbicycle_group": 162, # ???
# 	"bicycle_group": 163, # ???
# 	"person_group": 164, # ???
# 	"rider_group": 165, # ???
# 	"truck_group": 166, # ???
# 	"bus_group": 167, # ???
# 	"tricycle_group": 168, # ???
# 	"road": 49,
# 	"siderwalk": 50,
# 	"traffic_cone": 65,
# 	"road_pile": 66,
# 	"fence": 67,
# 	"traffic_light": 81,
# 	"pole": 82,
# 	"traffic_sign": 83,
# 	"wall": 84,
# 	"dustbin": 85,
# 	"billboard": 86,
# 	"building": 97,
# 	"bridge": 98,
# 	"tunnel": 99,
# 	"overpass": 100,
# 	"vegatation": 113,
# 	"unlabeled": 255
# }

# def intersect_kaggle_coco():
# 	with open('instances_val2017.json') as json_data:
# 		coco_meta = json.load(json_data)
# 	coco_categories = coco_meta['categories']
# 	intersection = []
# 	for category in coco_categories:
# 		if category['name'] in kaggle_categories:
# 			intersection.append(category)
# 			print(str(category['id']) + ': ' +
# 				str(kaggle_categories[category['name']]) +
# 				', #' + category['name'])
# 	return intersection
# intersect_kaggle_coco()

# missing tricycle
coco_to_kaggle = {
	1: 36, #person
	2: 35, #bicycle
	3: 33, #car
	4: 34, #motorcycle
	6: 39, #bus
	8: 38 #truck
}

def transform_coco_results(results):
	for i,result in enumerate(results):
		# mask to tell which instances are relevant for kaggle
		kaggle_idx = np.zeros(len(result['class_ids']), dtype=bool)
		for j,class_id in enumerate(result['class_ids']):
			if class_id in coco_to_kaggle:
				result['class_ids'][j] = coco_to_kaggle[class_id]
				kaggle_idx[j] = True
		# assert(np.sum(kaggle_idx) > 0)
		# 0-dim refers to instance in all cases below
		results[i]['class_ids'] = result['class_ids'][kaggle_idx]
		results[i]['rois'] = result['rois'][kaggle_idx]
		results[i]['masks'] = result['masks'][:,:,kaggle_idx].T # (N,W,H)
		results[i]['scores'] = result['scores'][kaggle_idx]
	return results

def run_length_encode(mask):
	'''
	x: numpy array of shape (width, height), 1 - mask, 0 - background
	Returns run length as list
	'''
	dots = np.where(mask.T.flatten()==1)[0]
	run_lengths = []
	prev = -2
	for b in dots:
		if b > prev+1:
			run_lengths.append([b, 0])
		run_lengths[-1][1] += 1
		prev = b
	run_str = ''
	for run in run_lengths:
		run_str += '{} {}|'.format(run[0], run[1])
	return run_str

def create_submit_csv_header():
        with open('submit_0.csv', 'w') as csvfile:
                csvwriter = csv.writer(csvfile,  delimiter=',')
                csvwriter.writerow( # header row
                        ['ImageId','LabelId','PixelCount','Confidence','EncodedPixels']
                )
def create_submit_csv(img_ids, kaggle_results):
	# img_ids: 			list of strings containing ImageId corresponding
	#					to each item in kaggle_results
	# kaggle_results:	result from coco processing, transformed for kaggle
	timenow = re.sub('\ |\-|\:', '_', str(datetime.datetime.now()))
	#with open('submit_'+timenow+'.csv', 'w') as csvfile:
	with open('submit_0.csv', 'a') as csvfile:
		csvwriter = csv.writer(csvfile,  delimiter=',')
		#csvwriter.writerow( # header row
		#	['ImageId','LabelId','PixelCount','Confidence','EncodedPixels']
		#)
		row_data = ['']*5 # data rows
		for i,result in enumerate(kaggle_results):
			for instance in range(len(result['class_ids'])):
				row_data[0] = img_ids[i]
				row_data[1] = result['class_ids'][instance]
				row_data[2] = np.sum(result['masks'][instance])
				row_data[3] = result['scores'][instance]
				row_data[4] = run_length_encode(result['masks'][instance])
				csvwriter.writerow(row_data)

#####################################################################
#####################################################################
##########################	demo submission	#########################
#####################################################################
#####################################################################
# with open('results.pkl', 'rb') as f:
# 	coco_results = pickle.load(f)

# kaggle_results = transform_coco_results(coco_results)
# print(kaggle_results[0].keys())
# print(kaggle_results[0]['class_ids'])
# print(kaggle_results[0]['masks'].shape)

# create_submit_csv(['1','2','3','4','5','6'],kaggle_results)











