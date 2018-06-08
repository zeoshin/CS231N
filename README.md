# CS231N
Group project for CS231N

mask-rcnn: the baseline architecture

mask-rcnn-tpn: mask rcnn with tpn added

mask-rcnn-tpn-spn: mask rcnn with tpn and spn added

We used coco API to read datasets. You need to download the Kaggle dataset from:
https://www.kaggle.com/c/cvpr-2018-autonomous-driving/data

Run command:
python coco_multi.py evaluate --dataset=dataset_path --year=2017 --model=pointer_to_your_weights_file