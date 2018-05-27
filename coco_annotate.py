import cv2
import numpy as np
import mask

filepath = './example_label.png'
img = cv2.imread(filepath, -1)

mask_list = []
for val in np.unique(img):
	if val != 255:
		mask_list.append(img == val)

labels_info = []
for mask in mask_list:
	# opencv 3.2
	mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8),
														cv2.RETR_TREE,
														cv2.CHAIN_APPROX_SIMPLE)
	segmentation = []

	for contour in contours:
		contour = contour.flatten().tolist()
		segmentation.append(contour)
		if len(contour) > 4:
			segmentation.append(contour)
	if len(segmentation) == 0:
		continue
	# get area, bbox, category_id and so on
	labels_info.append(
		{
			"segmentation": segmentation,  # poly
			# "area": area,  # segmentation area
			# "iscrowd": 0,
			# "image_id": index,
			# "bbox": [x1, y1, bbox_w, bbox_h],
			# "category_id": category_id,
			# "id": label_id
		},
	)

print(labels_info[0]['segmentation'][0])
print()
print(labels_info[0]['segmentation'][1])
print(len(labels_info))


