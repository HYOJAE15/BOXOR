import cv2
import sys

import numpy as np 

from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import csv

from utils_boxor.utils import *

import torch

from pathlib import Path

from ultralytics.utils.plotting import Annotator, colors

sys.path.append("./yolov5")
from utils.dataloaders import LoadImages
from models.common import DetectMultiBackend
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import (check_img_size, Profile, non_max_suppression, scale_boxes, scale_segments)
from utils.segment.general import process_mask, masks2segments

class AutoLabelButton :
    def __init__(self) :
        super().__init__()
                
    def read_image_label(self, path_to_img: str, path_to_txt: str, normilize: bool = True):
        # read image
        image = cv2.imread(path_to_img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
    
        # read .txt file for this image
        with open(path_to_txt, "r") as f:
            
            txt_file = f.readlines()
            
            for line in txt_file:
                line = line.split()
                cls_idx = int(line[0]) + 1
                coords = line[1:]
                polygon = np.array([[eval(x), eval(y)] for x, y in zip(coords[0::2], coords[1::2])]) # convert list of coordinates to numpy massive
                
                # Convert normilized coordinates of polygons to coordinates of image
                polygon[:,0] = polygon[:,0]*img_w
                polygon[:,1] = polygon[:,1]*img_h
                polygon = polygon.astype(np.int)
                
                # Fill the Ploygon label
                cv2.fillPoly(self.label, pts=[polygon], color=(cls_idx, cls_idx, cls_idx))

    def yoloDetection(self):
        # Inference Detection model
        self.yolo_result = self.yolo_det(self.img)
        self.yolo_result.print()
        result_json = self.yolo_result.pandas().xyxy[0].to_json(orient="records")
        print(result_json)
        
    def yoloSegmentation(self):
        # Inference Segmentation model
        ## class: {0: 'nusu', 1: 'baektae', 2: 'bakri', 3: 'bakrak', 4: 'kyunyeol', 5: 'cheolgeunnochul', 6: 'chungbunli', 7: 'kyunyeolbosu'}
        
        # Attribute
        conf_thres=0.25 # confidence threshold
        iou_thres=0.45 # NMS IOU threshold
        max_det=1000 # maximum detections per image
        device = select_device('')
        imgsz = (640, 640)
        classes = None
        agnostic_nms=False # class-agnostic NMS
        line_thickness=3 # bounding box thickness (pixels)
    
        # Load Model
        model = DetectMultiBackend(self.yolo_seg, device=device, dnn=False, data=None, fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        
        # Data Loader
        bs = 1
        dataset = LoadImages(self.imgPath, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = False
                pred, proto = model(im, augment=False, visualize=visualize)[:2]

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
            
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1

                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                
                if len(pred[0])>0:
        
                    # Segments
                    segments = [
                        scale_segments(im.shape[2:], x, im0.shape, normalize=True)
                        for x in reversed(masks2segments(masks))]

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    
                    self.txt_path = self.labelPath.replace(".png", "")
                    if os.path.isfile(f'{self.txt_path}.txt'):
                        os.remove(f'{self.txt_path}.txt')

                    # Write results
                    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                        line = (cls, *seg)  # label format
                        with open(f'{self.txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        
        if len(pred[0])>0:
            
            self.read_image_label(path_to_img=self.imgPath, 
                                  path_to_txt=f"{self.txt_path}.txt", 
                                  normilize=True)
            
            self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
            self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
            self.resize_image()
        else :
            print(f"No detection results")
        