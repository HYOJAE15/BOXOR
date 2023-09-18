
from cmath import e
import cv2
import sys

import numpy as np 
import matplotlib.pyplot as plt

from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import csv

from utils.utils import *

sys.path.append("./dnn/mmsegmentation")
from mmseg.apis import inference_segmentor



class AutoLabelButton :
    def __init__(self) :
        super().__init__()
                

    def roi256(self):
        print("roi256")
        self.brushButton.setChecked(False)
        self.roiAutoLabelButton.setChecked(True)
        self.getPointsButton.setChecked(False)

        self.use_brush = False

        self.use_erase = False

        self.set_roi = False
        
        self.set_roi_full = False

        self.set_roi_256 = True

        self.get_points_roi = False


    def roiRec(self):
        self.brushButton.setChecked(False)
        self.roiAutoLabelButton.setChecked(True)
        self.getPointsButton.setChecked(False)
        
        self.use_brush = False

        self.use_erase = False

        self.set_roi = True

        self.set_roi_full = False
        
        self.set_roi_256 = False

        self.get_points_roi = False

        
    def fullImg(self):
        self.brushButton.setChecked(False)
        self.roiAutoLabelButton.setChecked(True)
        self.getPointsButton.setChecked(False)
        
        self.use_brush = False

        self.use_erase = False

        self.set_roi = False

        self.set_roi_full = True
        
        self.set_roi_256 = False

        self.get_points_roi = False


    def getPoints(self):
        print("getPoints")
        self.get_points_roi = 1 - self.get_points_roi
        
        if self.get_points_roi:

            self.roiAutoLabelButton.setChecked(True)
            self.getPointsButton.setChecked(True)
        else :
            self.roiAutoLabelButton.setChecked(False)
            self.getPointsButton.setChecked(False)

        if self.set_roi :
            self.set_roi = False
        
        if self.set_roi_256 :
            self.set_roi_256 = False

        if self.use_erase : 
                self.use_erase = False
                self.eraseButton.setChecked(False)

        if  hasattr(self, 'eraseMenu'):   
            self.eraseMenu.close()

        if self.use_brush :
            self.use_brush = False
            self.brushButton.setChecked(False)
            
        if hasattr(self, 'brushMenu'):
            self.brushMenu.close()

        

    def getPointsRoi (self, event):
        print("getPointsRoi")
        try : 

            self.x_r256, self.y_r256 = getScaledPoint(event, self.scale)
            
            """
            NOTE: 256*256 Automatic Labeling on Getpoints Mode Calculation Coordinate

            FIXME: 이미지 가장자리쪽을 Auto Labeling 시 좌표를 이미지 shape 넘어서 받는다.
            TODO: 
            1. ROI가 이미지 좌표계를 넘어가는 경우 8가지와 박스가 이미지 내부에서 생성되는 경우 
                총 9가지 경우의 생각 이미지 좌표계는 (x, y) cv2로 읽으면 (y, x)
            2. np.clip으로 image shape크기로 제한을 주어서 해결
            """
            # 이미지 좌측 상단
            if self.x_r256 <= 128 and self.y_r256 <= 128 :
                print(f"이미지 좌측 상단")
                self.rect_start = [0, 0]
                self.rect_end = [self.x_r256+128, self.y_r256+128]
            
            # 이미지 좌측 (중앙, 하단)
            elif self.x_r256 < 128 and self.y_r256 > 128 :
                print(f"이미지 좌측 (중앙, 하단)")
                self.rect_start = [0, self.y_r256-128]
                # self.rect_end = self.x_r256+128, self.y_r256+128
                self.rect_end = [self.x_r256+128, self.y_r256+128] if self.img.shape[0]>=(self.y_r256+128) else [self.x_r256+128, self.img.shape[0]]
                print(f"클릭한 좌표: {self.x_r256, self.y_r256}")
                print(f"이미지 좌측 중단: {self.rect_end}")

            # 이미지 우측 (중앙, 하단)
            elif self.x_r256+128 > self.img.shape[1] and self.y_r256-128 >= 0 :
                print(f"이미지 우측 (중앙, 하단)") 
                self.rect_start = [self.x_r256-128, self.y_r256-128]
                self.rect_end = [self.img.shape[1], self.y_r256+128] if (self.y_r256+128)<self.img.shape[0] else [self.img.shape[1], self.img.shape[0]] 
                print(f"self.rect_end: {self.rect_end}")

            # 이미지 상단 (중앙, 우측)
            elif self.y_r256 < 128 and self.x_r256 > 128 :
                print(f"이미지 상단 (중앙, 우측)")
                self.rect_start = [self.x_r256-128, 0]
                # self.rect_end = self.x_r256+128, self.y_r256+128
                self.rect_end = [self.x_r256+128, self.y_r256+128] if self.img.shape[1]>=(self.x_r256+128) else [self.img.shape[1], self.y_r256+128] 
            
            # 이미지 내부, 이미지 하단 중앙
            else :
                print(f"이미지 내부, 이미지 하단 중앙")
                self.rect_start = [self.x_r256-128, self.y_r256-128]
                # self.rect_end = [self.x_r256+128, self.y_r256+128]
                self.rect_end = [self.x_r256+128, self.img.shape[0]] if (self.y_r256+128)>self.img.shape[0] else [self.x_r256+128, self.y_r256+128]
            

            """
            NOTE: Automatic Labeling by Deep Learning
            """
            src = self.src[self.rect_start[1]: self.rect_end[1], self.rect_start[0]: self.rect_end[0], :]
            
            result = inference_segmentor(self.model, src)

            idx = np.argwhere(result[0] == 1)
            y_idx, x_idx = idx[:, 0], idx[:, 1]
            x_idx = x_idx + self.rect_start[0]
            y_idx = y_idx + self.rect_start[1]

            self.label[y_idx, x_idx] = self.label_segmentation # label_palette 의 인덱스 색깔로 표현
            
            self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
            self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
            self.resize_image()

            
            """
            NOTE: ROI Box Computer Visualize
            TODO: 최근 시점은 mainwindow에서 보여주고 그전 시점들은 계속 쌓아가자
            """
            thickness = 2    

            rect_256 = cv2.rectangle(
                self.colormap.copy(), self.rect_start, self.rect_end, (255, 255, 255), thickness
            )
            self.pixmap = QPixmap(cvtArrayToQImage(rect_256))
            self.resize_image()


            # 중첩률 계산을 위한 공행렬 생성 및 중첩률 계산
            # uint (Unsigned int: 부호 없이 나타내는 정수... 양수라고 생각), uint8 (0~2^8: 0~255, 총 256개 정수) 
            # roi_coord_img = np.zeros(self.img.shape, np.uint8)
            
            
            self.roi_coord_img = np.zeros(self.img.shape)
            
            self.roi_union = np.zeros(self.img.shape[:2]) 
            
            self.roi_last = np.zeros(self.img.shape[:2])
            self.roi_last = cv2.rectangle(self.roi_last, self.rect_start, self.rect_end, (1, 1, 1), -1)
            
            
            """
            NOTE:좌표 저장 모드 시 csv 파일 생성과 Overlap Rate(OR) 측정
            
            FIXME: 마지막 좌표 저장 여부 확인 하고 고쳐라 아니면 해결 방안 이라도 ...
            VSD로 종료 시 마지막 좌표 저장 안됨, GUI창 닫힘 버튼을 눌러서 종료해야 마지막 좌표 저장  
            좌표 저장 에러 해결, 마지막 좌표 저장 여부 확인 해라 (모든 경우)
            이미지 이동 후 다른 파일로 csv 파일을 읽으면 마지막 좌표 저장 가능 
            """
            
            # Create csv file and save img coordinate, overlap rate
            self.saveFolderName = os.path.dirname(self.imgPath)
            self.saveFolderName = os.path.dirname(self.saveFolderName)
            self.saveFolderName = os.path.dirname(self.saveFolderName)
            self.saveFolderName = os.path.join(self.saveFolderName, "Coordinate")
            # 최상위 폴더(gtFine, leftImg8bit)
            self.saveImgName = os.path.basename(self.imgPath)
                    
            self.csvImgName = self.saveImgName.replace("_leftImg8bit.png", ".csv")
            os.makedirs(self.saveFolderName, exist_ok=True)
                        
            with open(os.path.join(self.saveFolderName, self.csvImgName), "a", encoding="cp949", newline="") as f :
                self.situationLabel.setText(self.csvImgName + "을(를) Coordinate 폴더에 저장하였습니다.")
            

            self.overlap_list = [[0,0,0,0]]
                          
            # overlap rate
            with open(os.path.join(self.saveFolderName, self.csvImgName), "r", encoding="cp949", newline="") as overlap :
                overlap_list = csv.reader(overlap)
                for line in overlap_list :
                    self.overlap_list.append(line)
                    
            for idx in self.overlap_list[1:]:   
                start=[int(idx[2]), int(idx[0])]
                end=[int(idx[3]), int(idx[1])]
                self.roi_coord_img = cv2.rectangle(self.roi_coord_img, start, end, (1, 1, 1), -1)
                self.roi_coord_img = cv2.rectangle(self.roi_coord_img, self.rect_start, self.rect_end, (0, 255, 0), -1)
                cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("ROI", int(self.img.shape[1]*0.5), int(self.img.shape[0]*0.5))
                cv2.imshow("ROI", self.roi_coord_img)
                
                self.roi_union = cv2.rectangle(self.roi_union, start, end, (1, 1, 1), -1)


            
            
            
            """
            현재 시점 ROI와 전 시점 ROI 합집합과의 교집합
            """
            union = np.count_nonzero(self.roi_union)
            print(f"union: {union}")
            last_roi = cv2.countNonZero(self.roi_last)
            print(f"last_roi: {last_roi}")
            if union > 0 : 
                intersection = cv2.bitwise_and(self.roi_last, self.roi_union)
                # cv2.namedWindow("intersection", cv2.WINDOW_NORMAL)
                # cv2.resizeWindow("intersection", int(self.img.shape[1]*0.5), int(self.img.shape[0]*0.5))
                # cv2.imshow("intersection", intersection)
                intersection_roi = cv2.countNonZero(intersection)
                print(f"intersection: {intersection_roi}")
                self.overlap_rate = intersection_roi/last_roi
                print(f"overlap_rate: {self.overlap_rate}")
            elif union == 0 :
                self.overlap_rate = 0
                print(f"union_0_overlap_rate: {self.overlap_rate}")

             

            # if cv2.countNonZero(self.roi_union) > 0 :
            #     overlap_rectangle = cv2.bitwise_and(self.roi_last, self.roi_union)
            #     overlap_rate = cv2.countNonZero(overlap_rectangle)
            #     print(f"overlap_rate: {overlap_rate}")
            #     plt.imshow(overlap_rectangle)
            #     plt.show()
            

            # overlap_rectangle = (abs(int(self.overlap_list[-1][3])-int(self.rect_start[0]))*abs(int(self.overlap_list[-1][1])-int(self.rect_start[1])))
            # print(f"overlap_rec: {overlap_rectangle}")
            # current_rectangle = abs(int(self.rect_end[0])-int(self.rect_start[0]))*abs(int(self.rect_end[1])-int(self.rect_start[1]))
            # print(f"current_rectangle: {current_rectangle}")

            
            # overlap_rate = overlap_rectangle/current_rectangle    
            # print(f"overlap_rate: {overlap_rate}")
            
            

            print(f"currentimgshape: {self.img.shape}")

            self.pointsList = [self.rect_start[1], self.rect_end[1],
                               self.rect_start[0], self.rect_end[0],
                               f"x: {self.x_r256}", f"y: {self.y_r256}",
                               f"classIdx:{self.label_segmentation}",
                               f"overlap rate:",
                               f"{self.overlap_rate}"]
            
            
            with open(os.path.join(self.saveFolderName, self.csvImgName), "a", encoding="cp949", newline="") as f :
                csvWriter = csv.writer(f)
                csvWriter.writerow(self.pointsList)

            
            """
            NOTE: Test
            """
            # plt.imshow(self.roi_coord_img)
            # plt.show()

            # cv2.imshow("rec", self.roi_coord_img)
            # # cv2.waitKey(0)
            
            
            


        except ZeroDivisionError as e :
            print(e)

    # Get Points set Rectangle roi  
    def GPRpress (self, event):
        print("GPRpress")
        x, y = getScaledPoint(event, self.scale)

        self.rect_start_GP = [x, y]
        print(f"rect_start: {self.rect_start_GP}")

    def GPRmove (self, event) :
        x, y = getScaledPoint(event, self.scale)

        self.rect_end_GP = [x, y]

        thickness = 5

        rect_hover = cv2.rectangle(
            self.colormap.copy(), self.rect_start_GP, self.rect_end_GP, (255, 255, 255), thickness) # , color, thickness, lineType,)
        self.pixmap = QPixmap(cvtArrayToQImage(rect_hover))
        self.resize_image()
    def GPRrelease (self, event) :
        print("GPRrelease")
        x, y = getScaledPoint(event, self.scale)
        print(f"x, y: {x},{y}")
    
        if x < self.rect_start_GP[0] : 
            temp = x 
            x = self.rect_start_GP[0]
            print(f"x: {x}")
            self.rect_start_GP[0] = temp
            print(f"rect_start: {self.rect_start_GP[0]}") 
            
        if y < self.rect_start_GP[1] : 
            temp = y 
            y = self.rect_start_GP[1]
            self.rect_start_GP[1] = temp 

        self.rect_end_GP = [x, y] 

        if (self.rect_end_GP[0] == self.rect_start_GP[0]) | (self.rect_end_GP[1] == self.rect_start_GP[1]):
            print(f"드래그 안함")
            self.rect_start_GP[0] -= 128
            self.rect_start_GP[1] -= 128  

            self.rect_end_GP[0] += 128 
            self.rect_end_GP[1] += 128 

        print(f"self.label.shape: {self.label.shape}")
        self.rect_start_GP[0] = np.clip(self.rect_start_GP[0], 0, self.label.shape[1])
        self.rect_start_GP[1] = np.clip(self.rect_start_GP[1], 0, self.label.shape[0])

        self.rect_end_GP[0] = np.clip(self.rect_end_GP[0], 0, self.label.shape[1])
        self.rect_end_GP[1] = np.clip(self.rect_end_GP[1], 0, self.label.shape[0])
            

        result = inference_segmentor(self.model, self.src[self.rect_start_GP[1]: self.rect_end_GP[1], self.rect_start_GP[0]: self.rect_end_GP[0], :])
        print(f"GPRrelease Coordinate: {self.rect_start_GP[1], self.rect_end_GP[1], self.rect_start_GP[0], self.rect_end_GP[0]}")
        idx = np.argwhere(result[0] == 1)
        y_idx, x_idx = idx[:, 0], idx[:, 1]
        x_idx = x_idx + self.rect_start_GP[0]
        y_idx = y_idx + self.rect_start_GP[1]

        self.label[y_idx, x_idx] = self.label_segmentation
        
        self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
        self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        self.resize_image()

        thickness = 2    

        rect_256 = cv2.rectangle(
                self.colormap.copy(), self.rect_start_GP, self.rect_end_GP, (255, 255, 255), thickness
            )
        self.pixmap = QPixmap(cvtArrayToQImage(rect_256))
        self.resize_image()


        self.roi_coord_img = np.zeros(self.img.shape)
            
        self.roi_union = np.zeros(self.img.shape[:2]) 
        
        self.roi_last = np.zeros(self.img.shape[:2])
        self.roi_last = cv2.rectangle(self.roi_last, self.rect_start_GP, self.rect_end_GP, (1, 1, 1), -1)
        

        # Create csv File

        print(f"autolabelScripts : {self.imgPath}" )
        self.saveFolderName = os.path.dirname(self.imgPath)
        self.saveFolderName = os.path.dirname(self.saveFolderName)
        self.saveFolderName = os.path.dirname(self.saveFolderName)
        self.saveFolderName = os.path.join(self.saveFolderName, "Coordinate")
        # 최상위 폴더(gtFine, leftImg8bit)
        self.saveImgName = os.path.basename(self.imgPath)
        self.csvImgName = self.saveImgName.replace("_leftImg8bit.png", ".csv")
        
        os.makedirs(self.saveFolderName, exist_ok=True)
        
        with open(os.path.join(self.saveFolderName, self.csvImgName), "a", encoding="cp949", newline="") as f :
            self.situationLabel.setText(self.csvImgName + "을(를) Coordinate 폴더에 저장하였습니다.")
        
        self.overlap_list = [[0,0,0,0]]
        
        # overlap rate
        with open(os.path.join(self.saveFolderName, self.csvImgName), "r", encoding="cp949", newline="") as overlap :
            overlap_list = csv.reader(overlap)
            for line in overlap_list :
                self.overlap_list.append(line)
                
        for idx in self.overlap_list[1:]:   
            start=[int(idx[2]), int(idx[0])]
            end=[int(idx[3]), int(idx[1])]
            self.roi_coord_img = cv2.rectangle(self.roi_coord_img, start, end, (1, 1, 1), -1)
            self.roi_coord_img = cv2.rectangle(self.roi_coord_img, self.rect_start_GP, self.rect_end_GP, (0, 255, 0), -1)
            cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ROI", int(self.img.shape[1]*0.5), int(self.img.shape[0]*0.5))
            cv2.imshow("ROI", self.roi_coord_img)
            
            self.roi_union = cv2.rectangle(self.roi_union, start, end, (1, 1, 1), -1)


        
        
        
        """
        현재 시점 ROI와 전 시점 ROI 합집합과의 교집합
        """
        union = np.count_nonzero(self.roi_union)
        print(f"union: {union}")
        last_roi = cv2.countNonZero(self.roi_last)
        print(f"last_roi: {last_roi}")
        if union > 0 : 
            intersection = cv2.bitwise_and(self.roi_last, self.roi_union)
            # cv2.namedWindow("intersection", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("intersection", int(self.img.shape[1]*0.5), int(self.img.shape[0]*0.5))
            # cv2.imshow("intersection", intersection)
            intersection_roi = cv2.countNonZero(intersection)
            print(f"intersection: {intersection_roi}")
            self.overlap_rate = intersection_roi/last_roi
            print(f"overlap_rate: {self.overlap_rate}")
        elif union == 0 :
            self.overlap_rate = 0
            print(f"union_0_overlap_rate: {self.overlap_rate}")

        
        
        
        
        self.pointsList = [self.rect_start_GP[1], self.rect_end_GP[1],
                           self.rect_start_GP[0], self.rect_end_GP[0],
                           f"x: ", f"y: ",
                           f"class idx:{self.label_segmentation}",
                           f"overlap rate:",
                           f"{self.overlap_rate}"]
        
        with open(os.path.join(self.saveFolderName, self.csvImgName), "a", encoding="cp949", newline="") as f :
            csvWriter = csv.writer(f)
            csvWriter.writerow(self.pointsList)




    def roi256PressPoint(self, event):

        try : 

            self.x_r256, self.y_r256 = getScaledPoint(event, self.scale)
            if self.x_r256 < 128 and self.y_r256 < 128 :
                self.rect_start = 0, 0
                self.rect_end = self.x_r256+128, self.y_r256+128
            elif self.x_r256 < 128 :
                self.rect_start = 0, self.y_r256-128
                self.rect_end = self.x_r256+128, self.y_r256+128
            elif self.y_r256 < 128 :
                self.rect_start = self.x_r256-128, 0
                self.rect_end = self.x_r256+128, self.y_r256+128 
            else :
                self.rect_start = self.x_r256-128, self.y_r256-128
                self.rect_end = self.x_r256+128, self.y_r256+128

            src = self.src[self.rect_start[1]: self.rect_end[1], self.rect_start[0]: self.rect_end[0], :]
            
            result = inference_segmentor(self.model, src)

            cv2.imshow("cropImage", src)
            

            print(f'cropImage.shape {self.img[self.rect_start[1]: self.rect_end[1], self.rect_start[0]: self.rect_end[0], :].shape}')
            

            idx = np.argwhere(result[0] == 1)
            y_idx, x_idx = idx[:, 0], idx[:, 1]
            x_idx = x_idx + self.rect_start[0]
            y_idx = y_idx + self.rect_start[1]

            self.label[y_idx, x_idx] = self.label_segmentation # label_palette 의 인덱스 색깔로 표현
            
            self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
            self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
            self.resize_image()


            thickness = 2    

            rect_256 = cv2.rectangle(
                self.colormap.copy(), self.rect_start, self.rect_end, (255, 255, 255), thickness)

            print(f"rectangle size {self.rect_start, self.rect_end}")
            self.pixmap = QPixmap(cvtArrayToQImage(rect_256))
            self.resize_image()
            
        except ZeroDivisionError as e :
            print(e)

    def roiFullPressPoint(self, event):

        try : 
            
            src = self.src
            result = inference_segmentor(self.model, src)

            idx = np.argwhere(result[0] == 1)
            y_idx, x_idx = idx[:, 0], idx[:, 1]
            
            self.label[y_idx, x_idx] = self.label_segmentation # label_palette 의 인덱스 색깔로 표현
            
            self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
            self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
            self.resize_image()


            
        except ZeroDivisionError as e :
            print(e)
        
    def roiPressPoint(self, event):

        x, y = getScaledPoint(event, self.scale)

        self.rect_start = [x, y]

    def roiMovingPoint(self, event):

        x, y = getScaledPoint(event, self.scale)

        self.rect_end = [x, y]

        thickness = 5

        rect_hover = cv2.rectangle(
            self.colormap.copy(), self.rect_start, self.rect_end, (255, 255, 255), thickness) # , color, thickness, lineType,)
        self.pixmap = QPixmap(cvtArrayToQImage(rect_hover))
        self.resize_image()
        

    def roiReleasePoint(self, event):

        x, y = getScaledPoint(event, self.scale)
    
        if x < self.rect_start[0] : 
            temp = x 
            x = self.rect_start[0]
            self.rect_start[0] = temp 
            
        if y < self.rect_start[1] : 
            temp = y 
            y = self.rect_start[1]
            self.rect_start[1] = temp 

        self.rect_end = [x, y] 

        if (self.rect_end[0] == self.rect_start[0]) | (self.rect_end[1] == self.rect_start[1]):
            self.rect_start[0] -= int(128/self.scale)
            self.rect_start[1] -= int(128/self.scale)  

            self.rect_end[0] += int(128/self.scale) 
            self.rect_end[1] += int(128/self.scale) 

        self.rect_start[0] = np.clip(self.rect_start[0], 0, self.label.shape[1])
        self.rect_start[1] = np.clip(self.rect_start[1], 0, self.label.shape[0])

        self.rect_end[0] = np.clip(self.rect_end[0], 0, self.label.shape[1])
        self.rect_end[1] = np.clip(self.rect_end[1], 0, self.label.shape[0])
            
        src = self.src[self.rect_start[1]: y, self.rect_start[0]: x, :]
        dst = histEqualization_hsv(src)

        result = inference_segmentor(self.model, src)

        idx = np.argwhere(result[0] == 1)
        y_idx, x_idx = idx[:, 0], idx[:, 1]
        x_idx = x_idx + self.rect_start[0]
        y_idx = y_idx + self.rect_start[1]

        self.label[y_idx, x_idx] = self.label_segmentation
        
        self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
        self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        self.resize_image()


    def pointsRoi(self, y_start, y_end, x_start, x_end):
        
        src = self.src[y_start: y_end, x_start: x_end, :]
        result = inference_segmentor(self.model, src)
        cv2.imshow("src", src)

        idx = np.argwhere(result[0] == 1)
        y_idx, x_idx = idx[:, 0], idx[:, 1]
        x_idx = x_idx + x_start
        y_idx = y_idx + y_start

        self.label[y_idx, x_idx] = self.label_segmentation 
        
        self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
        self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        self.resize_image()

        
    def pointsRoi_histEq_gr(self, y_start, y_end, x_start, x_end):
               
        src = self.src[y_start: y_end, x_start: x_end, :]
        dst = histEqualization_gr(src)
        result = inference_segmentor(self.model, dst)
        cv2.imshow("dst", dst)

        idx = np.argwhere(result[0] == 1)
        y_idx, x_idx = idx[:, 0], idx[:, 1]
        x_idx = x_idx + x_start
        y_idx = y_idx + y_start

        self.label[y_idx, x_idx] = self.label_segmentation 
        
        self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
        self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        self.resize_image()

    def pointsRoi_histEq_ycc(self, y_start, y_end, x_start, x_end):
               
        src = self.src[y_start: y_end, x_start: x_end, :]
        dst = histEqualization_ycc(src)
        result = inference_segmentor(self.model, dst)
        cv2.imshow("dst", dst)

        idx = np.argwhere(result[0] == 1)
        y_idx, x_idx = idx[:, 0], idx[:, 1]
        x_idx = x_idx + x_start
        y_idx = y_idx + y_start

        self.label[y_idx, x_idx] = self.label_segmentation 
        
        self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
        self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        self.resize_image()
    
    def pointsRoi_histEq_hsv(self, y_start, y_end, x_start, x_end):
               
        src = self.src[y_start: y_end, x_start: x_end, :]
        dst = histEqualization_hsv(src)
        result = inference_segmentor(self.model, dst)
        cv2.imshow("dst", dst)

        idx = np.argwhere(result[0] == 1)
        y_idx, x_idx = idx[:, 0], idx[:, 1]
        x_idx = x_idx + x_start
        y_idx = y_idx + y_start

        self.label[y_idx, x_idx] = self.label_segmentation 
        
        self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
        self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        self.resize_image()


        
    
