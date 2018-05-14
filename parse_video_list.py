import os
import glob
import time
import shutil
""" Change the paths before using the script """

def balance_video(project_data_path = "/Volumes/myDisk/kaggle/Segmentation/"):
	""" Make the video lists for cam 5 and cam 6 have the same length
		A new folder 'train_video_list_modified/' will be created
	"""
	""" input:
			string project_data_path:
				absolute path to the kaggle data
	"""
	train_video_list_path = project_data_path + "train_video_list/"
	train_video_list_path_modified = project_data_path + "train_video_list_modified/"
	verbose = False
	if not os.path.isdir(train_video_list_path_modified):
		os.mkdir(train_video_list_path_modified)
	for road in range(1,4):
		for video in range(1,21):
			file_to_glob = "road0" + str(road) + "_cam_*_video_" + str(video) + "_image_list_train.txt"
			file_list = glob.glob(train_video_list_path + file_to_glob)
			if (len(file_list) == 0):
				continue
			else:
				f_cam_5 = open(file_list[0])
				f_cam_6 = open(file_list[1])
				f_cam_5_modified_path = train_video_list_path_modified + file_list[0].split("/")[-1]
				f_cam_6_modified_path = train_video_list_path_modified + file_list[1].split("/")[-1]
				print("Start parsing:")
				print(file_list[0])
				print(file_list[1])
				f_cam_5_modified = open(f_cam_5_modified_path, "w")
				f_cam_6_modified = open(f_cam_6_modified_path, "w")
				#road01_ins\ColorImage\Record036\Camera 5\170908_065449018_Camera_5.jpg
				#road01_ins\Label\Record036\Camera 5\170908_065449018_Camera_5_instanceIds.png

				f_cam_5_lines = f_cam_5.readlines()
				f_cam_6_lines = f_cam_6.readlines()

				max_line_num = max(len(f_cam_5_lines), len(f_cam_6_lines))
				i = 0
				j = 0
				while i < len(f_cam_5_lines) or j < len(f_cam_6_lines):
					# forward fill first, then back fill
					if i >= len(f_cam_5_lines):
						f_cam_5_modified.write(f_cam_5_lines[-1])
						curr_cam_6_line = f_cam_6_lines[j]
						f_cam_6_modified.write(curr_cam_6_line)
						j += 1
						continue
					elif j >= len(f_cam_6_lines):
						f_cam_6_modified.write(f_cam_6_lines[-1])
						curr_cam_5_line = f_cam_5_lines[j]
						f_cam_5_modified.write(curr_cam_5_line)
						i += 1
						continue
					curr_cam_5_line = f_cam_5_lines[i]
					curr_cam_6_line = f_cam_6_lines[j]
					timestamp_cam_5 = int(curr_cam_5_line.split("\\")[4].split("_")[1])
					timestamp_cam_6 = int(curr_cam_6_line.split("\\")[4].split("_")[1])
					if verbose:
						print("Current cam 5 timestamp is {}".format(timestamp_cam_5))
						print("Current cam 6 timestamp is {}".format(timestamp_cam_6))
					if (timestamp_cam_5 < timestamp_cam_6):
						f_cam_6_modified.write(curr_cam_6_line)
						f_cam_5_modified.write(curr_cam_5_line)
						i += 1
					elif (timestamp_cam_5 > timestamp_cam_6):
						f_cam_6_modified.write(curr_cam_6_line)
						f_cam_5_modified.write(curr_cam_5_line)
						j += 1
					else:
						f_cam_6_modified.write(curr_cam_6_line)
						f_cam_5_modified.write(curr_cam_5_line)
						i += 1
						j += 1
				if i < max_line_num:
					while i < len(f_cam_5_lines):
						f_cam_5_modified.write(f_cam_5_lines[-1])
						i += 1
				elif j < max_line_num:
					while j < len(f_cam_6_lines):
						f_cam_6_modified.write(f_cam_6_lines[-1])
						j += 1
				f_cam_5.close()
				f_cam_6.close()
				f_cam_5_modified.close()
				f_cam_6_modified.close()

	# check length
	for road in range(1,4):
		for video in range(1,21):
			file_to_glob = "road0" + str(road) + "_cam_*_video_" + str(video) + "_image_list_train.txt"
			file_list = glob.glob(train_video_list_path + file_to_glob)
			if (len(file_list) == 0):
				continue
			else:
				f_cam_5 = open(file_list[0])
				f_cam_6 = open(file_list[1])
				f_cam_5_modified_path = train_video_list_path_modified + file_list[0].split("/")[-1]
				f_cam_6_modified_path = train_video_list_path_modified + file_list[1].split("/")[-1]

				f_cam_5_modified = open(f_cam_5_modified_path)
				f_cam_6_modified = open(f_cam_6_modified_path)

				f_cam_5_lines = f_cam_5.readlines()
				f_cam_6_lines = f_cam_6.readlines()
				f_cam_5_modified_lines = f_cam_5_modified.readlines()
				f_cam_6_modified_lines = f_cam_6_modified.readlines()

				if len(f_cam_5_modified_lines) != len(f_cam_6_modified_lines):
					# check if cam_5.txt and cam_6.txt have the same file length
					print(f_cam_5_modified_path)
					print(f_cam_6_modified_path)
					print("file length mismatch!")
					exit(1)
				if len(f_cam_5_modified_lines) < max(len(f_cam_5_lines), len(f_cam_6_lines)):
					# check if modified files have the same as or more rows than before
					print(f_cam_5_modified_path)
					print(f_cam_6_modified_path)
					print("file size smaller than original file!")
					exit(1)

# split into k parts
def slpit_to_k_parts(project_data_path = "/Volumes/myDisk/kaggle/Segmentation/", k = 4, overwrite = False):
	""" Split a single video list into k parts
		A new folder '{k}_parts/' will be created
	"""
	""" input:
			string project_data_path:
				absolute path to the kaggle data
			int k:
				number of parts
			boolean overwrite:
				if overwrite old folder
	"""

	train_video_list_path_modified = project_data_path + "train_video_list_modified/"
	year, mon, mday, hour, minute, sec, wday, yday, isdst = time.gmtime()
	#new_dir_path = project_data_path + str(year) + "_" + str(mon) + "_" + str(mday) + "_" + str(hour) + \
	#	"_" + str(minute) + "_" + str(sec) + "/"
	new_dir_path = project_data_path + str(k) + "_parts/"

	if overwrite and os.path.isdir(new_dir_path):
		try:
		    shutil.rmtree(new_dir_path)
		except OSError as e:
		    print ("Error: %s - %s." % (e.filename,e.strerror))

	if not os.path.isdir(new_dir_path):
		os.mkdir(new_dir_path)

		for road in range(1,4):
			for video in range(1,21):
				file_to_glob = "road0" + str(road) + "_cam_*_video_" + str(video) + "_image_list_train.txt"
				file_list = glob.glob(train_video_list_path_modified + file_to_glob)
				if (len(file_list) == 0):
					continue
				else:
					f_cam_5_modified = open(file_list[0])
					f_cam_6_modified = open(file_list[1])
					f_cam_5_modified_lines = f_cam_5_modified.readlines()
					f_cam_6_modified_lines = f_cam_6_modified.readlines()

					prev_part = 1
					f_cam_5_curr_part_path = new_dir_path + "part1" + "_road0" + str(road) + \
						"_cam_5_video_" + str(video) + "_image_list_train.txt"
					f_cam_6_curr_part_path = new_dir_path + "part1" + "_road0" + str(road) + \
						"_cam_6_video_" + str(video) + "_image_list_train.txt"

					f_cam_5_curr_part = open(f_cam_5_curr_part_path, "w")
					f_cam_6_curr_part = open(f_cam_6_curr_part_path, "w")

					for curr_line_idx in range(len(f_cam_5_modified_lines)):
						lines_per_part = len(f_cam_5_modified_lines) // k
						curr_part_idx = curr_line_idx // lines_per_part
						if prev_part != curr_part_idx + 1:
							f_cam_5_curr_part.close()
							f_cam_6_curr_part.close()

							f_cam_5_curr_part_path = new_dir_path + "part" + str(curr_part_idx + 1) + "_road0" + str(road) + \
								"_cam_5_video_" + str(video) + "_image_list_train.txt"
							f_cam_6_curr_part_path = new_dir_path + "part" + str(curr_part_idx + 1) + "_road0" + str(road) + \
								"_cam_6_video_" + str(video) + "_image_list_train.txt"

							f_cam_5_curr_part = open(f_cam_5_curr_part_path, "w")
							f_cam_6_curr_part = open(f_cam_6_curr_part_path, "w")

						f_cam_5_curr_part.write(f_cam_5_modified_lines[curr_line_idx])
						f_cam_6_curr_part.write(f_cam_6_modified_lines[curr_line_idx])

						if curr_line_idx == len(f_cam_5_modified_lines) - 1:
							f_cam_5_curr_part.close()
							f_cam_6_curr_part.close()

						prev_part = curr_part_idx + 1


if __name__ == "__main__":
	balance_video(project_data_path = "/Volumes/myDisk/kaggle/Segmentation/")
	slpit_to_k_parts(project_data_path = "/Volumes/myDisk/kaggle/Segmentation/", k = 4, overwrite = True)
