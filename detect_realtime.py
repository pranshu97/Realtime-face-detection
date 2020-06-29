import tensorflow as tf
from yolov3.yolov3 import Create_Yolov3
from yolov3.utils import load_yolo_weights, detect_realtime
from yolov3.configs import *

input_size = YOLO_INPUT_SIZE

yolo = Create_Yolov3(input_size=input_size, CLASSES=TRAIN_CLASSES)
yolo.load_weights("./checkpoints/yolov3_face_Tiny.h5")

out_path = './output1.mp4'
detect_realtime(yolo,out_path,show=True,CLASSES=TRAIN_CLASSES,iou_threshold=0.25)