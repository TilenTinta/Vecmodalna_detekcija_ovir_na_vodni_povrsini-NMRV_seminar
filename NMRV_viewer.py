import cv2, os, glob, json
import numpy as np
from matplotlib import pyplot as plt

# a script to display the NMRV seminar data
# i.e. RGB and thermal semantic segmentation labels
# and RGB and thermal detection bounding boxes

data_dir = './NMRV_seminar/'

def make_rgb_labels(labels, opencv_format=True):

	# takes a single channel image of semantic labels
	# outputs a colored 3-channel image for viewing/display

	classes_map = {'Static Obstacle': 1, 'Dynamic Obstacle': 2, 'Water': 3, 'Sky': 4, 'Water (hole)': 5, 'Ignore': 6, 'Recording Boat': 7}
	classes_colors = {'Static Obstacle': [0,1,0], 'Dynamic Obstacle': [1,0,0], 'Water': [0,0,1], 'Sky': [0,1,1], 'Water (hole)': [0,0,0], 'Ignore': [1,1,1], 'Recording Boat': [1,1,0]}

	labels_colored = np.zeros((labels.shape[0],labels.shape[1],3), dtype=np.uint8)
	for k,v in classes_map.items():
		if opencv_format:
			labels_colored[labels==v]=np.array(classes_colors[k][::-1])*255
		else:
			labels_colored[labels==v]=classes_colors[k]

	return labels_colored

def preprocess_thermal(thermal):
	thermal = cv2.cvtColor(thermal, cv2.COLOR_GRAY2RGB)
	thermal = cv2.normalize(thermal, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
	thermal = cv2.convertScaleAbs(thermal, alpha=255/(2**16))	

	return thermal

def view_rgb_segmentation():

	images = sorted(glob.glob(f'{data_dir}RGB_images/*'))
	annotations = sorted(glob.glob(f'{data_dir}RGB_semantic_annotations/*'))

	alpha = 0.5

	for im_fn, labels_fn in zip(images, annotations):

		im = cv2.imread(im_fn)
		labels = cv2.imread(labels_fn, cv2.IMREAD_UNCHANGED)
		labels = make_rgb_labels(labels)

		final = cv2.addWeighted(im, alpha, labels, 1-alpha, 0)

		cv2.imshow('final', final)

		key = cv2.waitKey(0)
		if key==27:
			return

def view_rgb_detection():

	with open(f"{data_dir}RGB_annotations.json", 'r') as f:
		annotations = json.load(f)

	images = sorted(glob.glob(f'{data_dir}RGB_images/*'))

	for im_fn in images:

		#name = im_fn.split('/')[-1][:-4]
		name = os.path.splitext(os.path.basename(im_fn))[0]


		im = cv2.imread(im_fn)	

		bboxes = annotations[name]
			
		for bbox in bboxes:
			cmin, rmin, cmax, rmax = bbox

			cv2.rectangle(im, (cmin,rmin),(cmax,rmax),(0,255,0),3)

		annotations[name]=bboxes

		cv2.imshow('im', im)

		key = cv2.waitKey(0)
		if key==27:
			return

def view_thermal_segmentation():
	images = sorted(glob.glob(f'{data_dir}thermal_images/*'))
	annotations = sorted(glob.glob(f'{data_dir}thermal_semantic_annotations/*'))

	alpha = 0.9

	for im_fn, labels_fn in zip(images, annotations):

		thermal = cv2.imread(im_fn, cv2.IMREAD_UNCHANGED)
		thermal = preprocess_thermal(thermal)

		labels = cv2.imread(labels_fn, cv2.IMREAD_UNCHANGED)
		labels = make_rgb_labels(labels)

		final = cv2.addWeighted(thermal, alpha, labels, 1-alpha, 0)

		cv2.imshow('final', final)

		key = cv2.waitKey(0)
		if key==27:
			return

def view_thermal_detection():

	with open(f"{data_dir}thermal_annotations.json", 'r') as f:
		annotations = json.load(f)

	images = sorted(glob.glob(f'{data_dir}thermal_images/*'))

	for im_fn in images:

		name = im_fn.split('/')[-1][:-4]

		thermal = cv2.imread(im_fn, cv2.IMREAD_UNCHANGED)
		thermal = preprocess_thermal(thermal)

		bboxes = annotations[name]
			
		for bbox in bboxes:
			cmin, rmin, cmax, rmax = bbox

			cv2.rectangle(thermal, (cmin,rmin),(cmax,rmax),(0,255,0),1)

		cv2.imshow('thermal', thermal)

		key = cv2.waitKey(0)
		if key==27:
			return

if __name__=='__main__':
	view_rgb_segmentation()
	view_rgb_detection()
	#view_thermal_segmentation()
	# view_thermal_detection()