import os
import numpy as np
import cv2

# video path examples
# road03_cam_5_video_7_image_list_train.txt
# road01_cam_6_video_17_image_list_test.txt

# image path example (from video .txt file)
# road01_ins\ColorImage\Record011\Camera 5\170908_061502408_Camera_5.jpg
# label path example (from video .txt file)
# road01_ins\Label\Record011\Camera 5\170908_061502408_Camera_5_instanceIds.png
# Extract filename and look for image and labels in:
# for training data: ./data/train_color/
# for training label: ./data/train_label/

# images (both data and label) have shape = (2710, 3384, 3)

train_vid_fold = './train_video_list/'
train_img_data_fold = './data/train_color/'
train_img_label_fold = './data/train_label/'

def organize_data(list_path):

	list_fnames = os.listdir(list_path)
	road_dict = {} # road -> video -> camera

	for fname in list_fnames:
		fname_split = fname.split('_')
		road = fname_split[0]
		cam = fname_split[2]
		vid = fname_split[4]
		if road not in road_dict:
			road_dict[road] = {}
		vid_dict = road_dict[road]
		if vid not in vid_dict:
			vid_dict[vid] = {}
		cam_dict = vid_dict[vid]
		cam_dict[cam] = list_path + fname

	return road_dict

def import_train_vid(vid_path):

	X,Y = None, None
	with open(vid_path) as f:
		for line in f:
			img_fname = line.split('\t')[0].split('\\')[-1].split()[0]
			img_path = train_img_data_fold + img_fname
			lbl_fname = line.split('\t')[1].split('\\')[-1].split()[0]
			lbl_path = train_img_label_fold + lbl_fname
			x = np.expand_dims(import_img(img_path),axis=0)
			y = np.expand_dims(import_img(lbl_path),axis=0)
			if X is None:
				X = x
				Y = y
			else:
				X = np.concatenate((X,x), axis=0)
				X = np.concatenate((Y,y), axis=0)
			break # DEBUG - get rid of this line during actual import

	return X,Y

def import_img(img_path):
	img = cv2.imread(img_path)
	assert(img is not None)
	return img


train_vid_dict = organize_data(train_vid_fold)
# print keys to dictionary layers (road -> video -> camera)
print(train_vid_dict.keys())
print(train_vid_dict['road02'].keys())
print(train_vid_dict['road02']['2'].keys())
print()

X,Y = import_train_vid(train_vid_dict['road01']['2']['5'])
# print shape of imported data (N, H, W, C)
print(X.shape)
print(Y.shape)
print()

x = X[0,:,:,:]
# print 10x10 crop of example image
offset_x, offset_y = 1000, 2000
cv2.imshow('image',x[offset_x:offset_x+10,offset_y:offset_y+10,:])
cv2.waitKey(0)
cv2.destroyAllWindows()










