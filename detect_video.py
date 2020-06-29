import tensorflow as tf
from yolov3.yolov3 import Create_Yolov3
from yolov3.utils import load_yolo_weights, detect_video
from yolov3.configs import *
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--video',type=str,default='IMAGES/test1.mp4',help='video path')
	parser.add_argument('--out',type=str,default='./',help='output path')
	args = parser.parse_args()

	input_size = YOLO_INPUT_SIZE

	yolo = Create_Yolov3(input_size=input_size, CLASSES=TRAIN_CLASSES)
	yolo.load_weights("./checkpoints/yolov3_face_Tiny.h5")

	vid_path = args.video
	out_path = args.out + 'output.mp4'
	detect_video(yolo,vid_path,out_path,show=True,CLASSES=TRAIN_CLASSES,iou_threshold=0.25)

if __name__ == '__main__':
	main()