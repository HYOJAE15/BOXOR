import sys
import cv2

import json
import os
import csv


import numpy as np 

from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


from scipy import ndimage

from utils_boxor.utils import *


from components.actions.actionFile import ActionFile

from components.buttons.autoLabelButton import AutoLabelButton
from components.buttons.brushButton import BrushButton
from components.buttons.eraseButton import EraseButton
from components.buttons.zoomButton import ZoomButton

from components.dialogs.brushMenuDialog import BrushMenu
from components.dialogs.eraseMenuDialog import EraseMenu
from components.dialogs.newProjectDialog import newProjectDialog
from components.dialogs.setCategoryDialog import setCategoryDialog

from components.opener.dialogOpener import dialogOpener

from components.widgets.treeView import TreeView

from components.dnnModel.damage import DL_Model

import torch

sys.path.append("./dnn/mmsegmentation")
from mmseg.apis import init_segmentor, inference_segmentor

import time

# Select folder "autolabel"
# MainWindow UI
project_ui = '../../ui_design/mainWindow.ui'

form = resource_path(project_ui)
form_class_main = uic.loadUiType(form)[0]

# Mainwindow class

class MainWindow(QMainWindow, form_class_main,
                 AutoLabelButton, BrushButton, EraseButton,
                 dialogOpener, 
                 ActionFile, TreeView) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        #### Attributes #### 
        
        # self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        self.brushSize = 2
        self.eraseSize = 2
        self.ver_scale = 1
        self.hzn_scale = 1
        self.x = 0 
        self.y = 0 
        self.label_class = 0
        self.DL_class = 0
        self.label_segmentation = 1
        self.alpha = 0.5
        self.use_brush = False
        self.use_erase = False
        self.set_roi = False
        self.set_roi_256 = False
        self.set_roi_full = False
        # get_points_roi Mode set_roi_256
        self.get_points_roi = False
        # get_points_roi Mode set_roi
        self.get_points_roi_setRec = False
        self.stopwatch = False
        self.circle = True
        
        # Yolo 모델 로드
        self.yolo_det = torch.hub.load('./yolov5', 'custom', path='./dnn/checkpoints/yolo_detection/yolo_det.pt', source='local', force_reload=True)
        # self.yolo_seg = torch.hub.load('./yolov5', 'custom', path='./dnn/checkpoints/yolo_segmentation/yolo_seg.pt', source='local', force_reload=True)
        self.yolo_seg = './dnn/checkpoints/yolo_segmentation/yolo_seg.pt'
        
        
        # 리스트위젯 에서 클릭된 클래스로 checkpoint 파일 불러오자 

        config_file = './dnn/checkpoints/2022.01.06 cgnet general crack 2048/cgnet_2048x2048_60k_CrackAsCityscapes.py'
        checkpoint_file = './dnn/checkpoints/2022.01.06 cgnet general crack 2048/iter_60000.pth'
        self.model = init_segmentor(config_file, 
                                    checkpoint_file, 
                                    device='cuda:0')


        
        # treeview setting 
        self.openFolderPath = None
        self.imgPath = None
        self.folderPath = None
        self.pathRoot = QtCore.QDir.rootPath()
        self.treeModel = QFileSystemModel(self)
        self.dialog = QFileDialog()   # Find the Folder or File Dialog
        self.treeView.clicked.connect(self.treeViewImage)
        self.treeView.clicked.connect(self.askSave)
        # self.treeView.keyPressEvent.connect(self.pressKey)
        
        # 1. Menu
        self.actionOpenFolder.triggered.connect(self.actionOpenFolderFunction)
        self.actionAddNewImages.triggered.connect(self.addNewImages)
        self.actionNewProject.triggered.connect(self.createNewProjectDialog)
        self.actionOpenProject.triggered.connect(self.openExistingProject)
        # self.actionCreate_a_Project.triggered.connect(self.openCreateProjectDialog)

        # 2. Zoom in and out
        self.ControlKey = False
        self.scale = 1
        
        # 3. brush & erase tools
        self.brushButton.clicked.connect(self.openBrushDialog)
        self.eraseButton.clicked.connect(self.openEraseDialog)

        # 4. main Image Viewer
        self.mainImageViewer.mousePressEvent = self.mousePressEvent
        self.mainImageViewer.mouseMoveEvent = self.mouseMoveEvent
        self.mainImageViewer.mouseReleaseEvent = self.mouseReleaseEvent
        self.mainImageViewer.wheelEvent = self.storeXY

        self.scrollArea.wheelEvent = self.wheelEventScroll


        # 5. listWidget
        self.listWidget.itemClicked.connect(self.getListWidgetIndex)
        self.DL_listWidget.itemClicked.connect(self.getDlListWidgetIndex)

        
        # 7. auto label tools 
        self.roiMenu = QMenu()
        self.roiMenu.addAction("256*256", self.roi256)
        self.roiMenu.addAction("Set Rectangle", self.roiRec)
        self.roiMenu.addAction("Full Image", self.fullImg)
        self.roiAutoLabelButton.setMenu(self.roiMenu)
        self.roiAutoLabelButton.clicked.connect(self.showRoiMenu)

        self.getPointsButton.clicked.connect(self.getPoints)
        self.yoloDetectButton.clicked.connect(self.yoloDetection)
        #self.roiAutoLabelButton.clicked.connect(self.runRoiAutoLabel)
    
        # 8. handMoveTool
        self.hKey = False
        self.icon = QPixmap("./Icon/square.png")
        self.scaled_icon = self.icon.scaled(QSize(5, 5), Qt.KeepAspectRatio)
        self.custom_cursor = QCursor(self.scaled_icon)

        
    
    def storeXY(self, event):
        if self.ControlKey:
            self.img_v_x = event.pos().x()
            self.img_v_y = event.pos().y()
            

    #### Methods ##### 

    ######################## 
    ### Image Processing ###
    ########################


    def addNewImages(self):
        
        try :

            if self.openFolderPath :
                self.imgPath = self.openFolderPath
                print(self.openFolderPath)
                print(self.imgPath)

            else :
                print(f'dang {self.imgPath}')
                # self.imgPath = self.openFolderPath
                # print(f"cityscapedataset 비준수 {self.openFolderPath}")

            readFilePath = self.dialog.getOpenFileNames(
                caption="Add images to current working directory", filter="Images (*.png *.jpg *.tiff)"
                )
            images = readFilePath[0]


                # check if images are from same folder
            if self.treeModel.rootPath() in os.path.dirname(images[0]):
                print("same foler")
                return None

            if self.imgPath :

                dotSplit_imgPath = self.imgPath.split(".")
                slashSplit_imgPath = self.imgPath.split("/")
              
                    # clicked img_file
                if 'png' in dotSplit_imgPath and 'leftImg8bit' in slashSplit_imgPath :

                    img_save_folder = os.path.dirname(self.imgPath)
                   
                    img_label_folder = os.path.dirname(self.labelPath)

                    print("png, left")
                
                    # clicked img_folder
                elif 'png' not in dotSplit_imgPath and 'leftImg8bit' in slashSplit_imgPath :
    
                    img_save_folder = self.imgPath
                    img_save_folder = img_save_folder.replace( '_leftImg8bit.png', '')  
                
                    img_label_folder = img_save_folder.replace('/leftImg8bit/', '/gtFine/')
                    img_label_folder = img_label_folder.replace( '_leftImg8bit.png', '')
                    print('left')

                else :   # 선택된 폴더가 시티스케이프 데이터셋 이 아닌 다른 경우 에러 발생 UnboundLocalError
                    print('not cityscapeDataset')
    
                for img in images:
                
                    temp_img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

                    img_filename = os.path.basename(img) # -> basename is file name
                    img_filename = img_filename.replace(' ', '')
                    img_filename = img_filename.replace('.jpg', '.png')
                    img_filename = img_filename.replace('.JPG', '.png')
                    img_filename = img_filename.replace('.tiff', '.png')
                    img_filename = img_filename.replace('.png', '_leftImg8bit.png')

                    img_gt_filename = img_filename.replace( '_leftImg8bit.png', '_gtFine_labelIds.png')
                    gt = np.zeros((temp_img.shape[0], temp_img.shape[1]), dtype=np.uint8)

                    is_success, org_img = cv2.imencode(".png", temp_img)
                    org_img.tofile(os.path.join(img_save_folder, img_filename))

                    is_success, gt_img = cv2.imencode(".png", gt)
                    gt_img.tofile(os.path.join(img_label_folder, img_gt_filename))

                    # check file extension -> change extension to png 
                    # create corresponding label file 

                    print(f'7 {os.path.join(img_save_folder, img_filename)}')

            else :
                print("self.imgPath is None")
        
        except IndexError as e :
            print(e)

        except UnboundLocalError as e :
            print(e)


    def updateLayers(self, x, y):
        try : 
            if self.use_brush :
                self.layers[self.label_class][y, x] = 1

            elif self.use_erase :
                self.layers[self.label_class][y, x] = 0
            
        except BaseException as e : 
            print(e)
        

    def updateLabelFromLayers(self, x, y):
        self.label[y, x] = 0
        temp_label = self.label[y, x]
        for lay_idx in reversed(range(1, len(self.layers))): 
            temp_label = np.where(self.layers[lay_idx][y, x], lay_idx, temp_label) 
        self.label[y, x] = temp_label

        
    def updateColormapFromLabel(self, x, y):
        try :             
            self.colormap[y, x] = self.img[y, x] * self.alpha + self.label_palette[self.label[y, x]] * (1-self.alpha)
            self.colormap[y, x] = blendImageWithColorMap(self.img[y, x], self.label[y, x], self.label_palette, self.alpha)
            self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        except BaseException as e : 
            print(e)
        

    def updateLabelandColormap(self, x, y):
        
        if self.use_brush :
            x, y = self.applyBrushSize(x, y)
        elif self.use_erase :
            x, y = self.applyEraseSize(x, y)


        try : 
            print(f"label_class {self.label_class}")
            print(type(self.label_class))
            if self.use_brush :
                self.label[y, x] = self.label_class
                self.colormap[y, x] = self.img[y, x] * self.alpha + self.label_palette[self.label_class] * (1-self.alpha)

            elif self.use_erase :
                self.label[y, x] = 0
                print("eraseMode")
                self.colormap[y, x] = self.img[y, x] * self.alpha + self.label_palette[0] * (1-self.alpha)

            
            self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        except BaseException as e : 
            print(e)

    


    def openExistingProject(self):

        try :

            readFilePath = self.dialog.getOpenFileName(
                caption="Select Project File", filter="*.hdr"
                )
            hdr_path = readFilePath[0]
            
            folderPath = os.path.dirname(hdr_path)
            print(folderPath)
            cityscapeDataset_folderPath = os.path.join(folderPath, "leftImg8bit")
                # openFolderPath 를 None 으로 받고 treeView 에서 선택한 파일 또는 폴더 주소를 받는다.
            self.openFolderPath = None
            print(os.path.join(folderPath, "leftImg8bit"))
            self.fileNameLabel.setText(cityscapeDataset_folderPath)
            self.treeModel.setRootPath(os.path.join(folderPath, 'leftImg8bit'))
            self.indexRoot = self.treeModel.index(self.treeModel.rootPath())
            self.treeView.setModel(self.treeModel)
            self.treeView.setRootIndex(self.indexRoot)
            

            with open(hdr_path) as f:
                hdr = json.load(f)

            self.listWidget.clear()

            self.label_palette = []

            for idx, cat in enumerate(hdr['categories']):
                name, color = cat[0], cat[1]
                color = json.loads(color)
                self.listWidget.addItem(name)
                iconPixmap = QPixmap(20, 20)
                iconPixmap.fill(QColor(color[0], color[1], color[2]))
                self.listWidget.item(idx).setIcon(QIcon(iconPixmap))
                self.label_palette.append(color)

            self.label_palette = np.array(self.label_palette)

        except FileNotFoundError as e:
            print(e)


    def createNewProjectDialog(self, event):
            # new_project_info 를 딕셔너리 자료형으로 설정 한다.
        self.new_project_info = {}

        self.newProjectDialog = newProjectDialog()
            # textProjectName : QTextEdit
        self.newProjectDialog.textProjectName.textChanged.connect(self.setProjectName)
        self.newProjectDialog.nextButton.clicked.connect(self.openCategoryInfoDialog)
        self.newProjectDialog.folderButton.clicked.connect(self.setFolderPath)

        self.newProjectDialog.exec()
        

    def setProjectName(self):
            # 딕셔너리 자료형으로 설정한 변수 에서 key 를 설정해주고 해당 key 에 value 값을 할당 한다.
            # self.new_project_info = {'project_name': self.newProjectDialog.textProjectName.toPlainText() }
        self.new_project_info['project_name'] = self.newProjectDialog.textProjectName.toPlainText()
        print(self.new_project_info['project_name'])


    def setFolderPath(self):

        readFolderPath = self.dialog.getExistingDirectory(None, "Select Folder", "./")
        print(readFolderPath)
            # folderPath : QTextEdit
        self.newProjectDialog.folderPath.setMarkdown(readFolderPath)
        self.new_project_info['folder_path'] = readFolderPath
        print(self.new_project_info)


    def createProjectHeader(self):

        createProjectFile_name = self.new_project_info['project_name'] + ".hdr"
        print(createProjectFile_name)

        path = self.new_project_info['folder_path']
        n_row = self.setCategoryDialog.tableWidget.rowCount()

        self.new_project_info['categories'] = []

        for i in range(n_row):
            self.new_project_info['categories'].append(
                [
                    self.setCategoryDialog.tableWidget.item(i, 0).text(),
                    self.setCategoryDialog.tableWidget.item(i, 2).text()
                ]
                )
            
        with open(os.path.join(path, createProjectFile_name), 'w') as fp:
            json.dump(self.new_project_info, fp)
            self.setCategoryDialog.close()


    def mousePressEvent(self, event):
        # print("mousePressEvent")

        if self.hKey : 
            self.scrollAreaMousePress(event)

        elif self.use_brush : 
            self.brushPressOrReleasePoint(event)

        elif self.use_erase :
            self.erasePressOrReleasePoint(event)

        elif self.set_roi : 
            self.roiPressPoint(event)

        elif self.set_roi_256 :
            self.roi256PressPoint(event)

        elif self.set_roi_full :
            self.roiFullPressPoint(event)

        elif self.get_points_roi and self.get_points_roi_setRec == False :
            self.getPointsRoi(event)

        elif self.get_points_roi and self.get_points_roi_setRec == True :
            self.GPRpress(event)


    def mouseMoveEvent(self, event):

        if self.hKey : 
            self.scrollAreaMouseMove(event)

        elif self.use_brush : 
            self.brushMovingPoint(event)

        elif self.use_erase :
            self.eraseMovingPoint(event)

        elif self.set_roi : 
            self.roiMovingPoint(event)

        elif self.get_points_roi and self.get_points_roi_setRec == True :
            self.GPRmove(event)

    def mouseReleaseEvent(self, event): 

        if self.hKey :
            pass

        elif self.use_brush : 
            self.brushPressOrReleasePoint(event)

        elif self.use_erase :
            self.erasePressOrReleasePoint(event)

        elif self.set_roi : 
            self.roiReleasePoint(event)
        
        elif self.get_points_roi and self.get_points_roi_setRec == True :
            self.GPRrelease(event)
    
    def showRoiMenu(self):
        self.roiAutoLabelButton.showMenu()

    def setVerticalScale(self, new_scale):
        self.ver_scale = new_scale

    def setHorizontalScale(self, new_scale):
        self.hzn_scale = new_scale

        # key press 에서 기능 을 키고 끄는것은 어떻게 하나
        # turn on : 단축키 press 
        # turn off : 한번더 press 
    def keyPressEvent(self, event):
        print(event.key())
            # zoom
        if event.key() == Qt.Key_Control:
            self.ControlKey = True
            # handMove
            # h_key 한번 press 후 마우스 로 이동 
        elif event.key() == Qt.Key_H: 
            self.hKey = True
            print(QCursor().shape())
            QApplication.setOverrideCursor(Qt.OpenHandCursor)

        
        
        # get_points_roi가 1이라면 A, S 에 자동 손상 탐지 라벨링 기능 추가
        elif event.key() == 65 : # A Key
            
            # get_points_roi 활성화 
            if self.get_points_roi == True :
                
                self.get_points_roi_setRec = False
                self.auto_256.setChecked(True)
                self.auto_sr.setChecked(False)
            
            
            # get_points_roi 비활성화
            elif self.get_points_roi == False :

                
                self.set_roi_256 = 1-self.set_roi_256

                if self.set_roi_256 : 
                    self.roiAutoLabelButton.setChecked(True)

                else : 
                    self.roiAutoLabelButton.setChecked(False)

                if self.set_roi :
                    self.set_roi = False

                if self.get_points_roi :
                    self.get_points_roi = False
                    self.getPointsButton.setChecked(False)


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

            
        elif event.key() == 89 : # Y key 
            print(f"self.se_seconds: ")
            

        # stopwatch tools
        elif event.key() == 84 : # T Key 
            # print("stopwatch")
            # self.stopwatch = 1 - self.stopwatch
            # print(f"stopwatch: {self.stopwatch}")
            
            if self.stopwatch == True :
                self.stopwatch = False
                self.stopwatchButton.setChecked(False)
                self.openStopwatchDialog(event)
                

            elif self.stopwatch == False :
                self.stopwatch = True
                self.stopwatchButton.setChecked(True)
                self.openStopwatchDialog(event)

        
        elif event.key() == 66 : # B Key
            print("B")

            if self.use_brush == True :
                self.use_brush = False
                self.brushButton.setChecked(False)

                if hasattr(self, 'brushMenu'):
                    self.brushMenu.close()  

            else :
                self.openBrushDialog(event)
                # self.listWidget.setCurrentRow(self.label_class)
                
            if self.use_erase : 
                self.use_erase = False
                self.eraseButton.setChecked(False)
                
            if  hasattr(self, 'eraseMenu'):   
                self.eraseMenu.close()
                
            if self.set_roi_256:
                self.set_roi_256 = False
                self.roiAutoLabelButton.setChecked(False)

            if self.set_roi:
                self.set_roi = False

            if self.get_points_roi :
                self.get_points_roi = False
                self.getPointsButton.setChecked(False)

        elif event.key() == 69 : # E Key
            print("E")

            if self.use_erase == True :
                self.use_erase = False
                self.eraseButton.setChecked(False)

                if  hasattr(self, 'eraseMenu'):   
                    self.eraseMenu.close()

            else :
                self.openEraseDialog(event)

            if self.use_brush :
                self.use_brush = False
                self.brushButton.setChecked(False)

            if hasattr(self, 'brushMenu'):
                self.brushMenu.close()

            if self.set_roi_256:
                self.set_roi_256 = False
                self.roiAutoLabelButton.setChecked(False)

            if self.set_roi:
                self.set_roi = False

            if self.get_points_roi :
                self.get_points_roi = False
                self.getPointsButton.setChecked(False)
            

        elif event.key() == 70 : # f Key
            print("filling works")
            
            self.layers = createLayersFromLabel(self.label, len(self.label_palette))
            self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
                
            
            self.layers[self.label_class] = ndimage.binary_fill_holes(self.layers[self.label_class])

            for idx in reversed(range(1, len(self.layers))): 
                self.label = np.where(self.layers[idx], idx, self.label) 

            self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
                
            self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))

            self.resize_image()
            
        # get_points_roi 활성화, 비활성화 모드 2개 나누어 생각
        # get_points_roi 활성화 시 기본으로 set_roi_256
        # get_points_roi 활성화 후 set_roi 모드 생각
        elif event.key() == 71 : # G key

            print("Get points")
            self.get_points_roi = 1-self.get_points_roi

            if self.get_points_roi : 
                self.roiAutoLabelButton.setChecked(True)
                self.getPointsButton.setChecked(True)
                self.auto_256.setChecked(True)
                self.auto_sr.setChecked(False)

            else : 
                self.roiAutoLabelButton.setChecked(False)
                self.getPointsButton.setChecked(False)
                self.auto_256.setChecked(False)
                self.auto_sr.setChecked(False)
                self.get_points_roi_setRec = False

            if self.set_roi_256 :
                self.set_roi_256 = False

            if self.set_roi :
                self.set_roi = False
                

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
            
        elif event.key() == 74: # j key
            
            saveFolderName = os.path.dirname(self.imgPath)
            saveFolderName = os.path.dirname(saveFolderName)
            saveFolderName = os.path.dirname(saveFolderName)
            saveFolderName = os.path.join(saveFolderName, "Coordinate")
            saveImgName = os.path.basename(self.imgPath)
                    
            csvImgName = saveImgName.replace("_leftImg8bit.png", ".csv")
            
            self.getPointsList = []

            with open(os.path.join(saveFolderName, csvImgName), "r", encoding="cp949", newline='') as f :
                data = csv.reader(f)
                for row in data:
                    self.getPointsList.append(row)

            for idx in self.getPointsList:
                AutoLabelButton.pointsRoi(self, int(idx[0]), int(idx[1]), int(idx[2]), int(idx[3]))
                
            """
            저장된 좌표의 탐지 결과와 ROI도 같이 보고 싶을때 사용
            """
            # 자동 탐지 라벨링 시 blendImageWithColorMap 함수를 통과 하면서
            # cv2.rectangle 했던 이미지가 없어짐, 블렌딩시 gt와 left 만 섞어서 쓰기때문
            for idx in self.getPointsList:
                rect_start = [int(idx[2]), int(idx[0])]
                rect_end = [int(idx[3]), int(idx[1])]

                thickness = 2    

                # 변수가 계속 살아남게 하여 이미지에 rect가 계속 쌓이는 형식
                self.colormap = cv2.rectangle(
                    self.colormap, rect_start, rect_end, (255, 255, 255), thickness)

                # print(f"rectangle size {rect_start, rect_end}")
                self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
                self.resize_image()
                
                
        elif event.key() == 76: # l key (gr)
            
            saveFolderName = os.path.dirname(self.imgPath)
            saveFolderName = os.path.dirname(saveFolderName)
            saveFolderName = os.path.dirname(saveFolderName)
            saveFolderName = os.path.join(saveFolderName, "Coordinate")
            saveImgName = os.path.basename(self.imgPath)
                    
            csvImgName = saveImgName.replace("_leftImg8bit.png", ".csv")
            
            self.getPointsList = []

            with open(os.path.join(saveFolderName, csvImgName), "r", encoding="cp949", newline='') as f :
                data = csv.reader(f)
                for row in data:
                    self.getPointsList.append(row)

            for idx in self.getPointsList:
                AutoLabelButton.pointsRoi_histEq_gr(self, int(idx[0]), int(idx[1]), int(idx[2]), int(idx[3]))
                
            """
            저장된 좌표의 탐지 결과와 ROI도 같이 보고 싶을때 사용
            """
            # 자동 탐지 라벨링 시 blendImageWithColorMap 함수를 통과 하면서
            # cv2.rectangle 했던 이미지가 없어짐, 블렌딩시 gt와 left 만 섞어서 쓰기때문
            for idx in self.getPointsList:
                rect_start = [int(idx[2]), int(idx[0])]
                rect_end = [int(idx[3]), int(idx[1])]

                thickness = 2    

                # 변수가 계속 살아남게 하여 이미지에 rect가 계속 쌓이는 형식
                self.colormap = cv2.rectangle(
                    self.colormap, rect_start, rect_end, (255, 255, 255), thickness)

                # print(f"rectangle size {rect_start, rect_end}")
                self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
                self.resize_image()
                
                
        elif event.key() == 78: # n key (hsv)
            
            saveFolderName = os.path.dirname(self.imgPath)
            saveFolderName = os.path.dirname(saveFolderName)
            saveFolderName = os.path.dirname(saveFolderName)
            saveFolderName = os.path.join(saveFolderName, "Coordinate")
            saveImgName = os.path.basename(self.imgPath)
                    
            csvImgName = saveImgName.replace("_leftImg8bit.png", ".csv")
            
            self.getPointsList = []

            with open(os.path.join(saveFolderName, csvImgName), "r", encoding="cp949", newline='') as f :
                data = csv.reader(f)
                for row in data:
                    self.getPointsList.append(row)

            for idx in self.getPointsList:
                AutoLabelButton.pointsRoi_histEq_hsv(self, int(idx[0]), int(idx[1]), int(idx[2]), int(idx[3]))
                
            """
            저장된 좌표의 탐지 결과와 ROI도 같이 보고 싶을때 사용
            """
            # 자동 탐지 라벨링 시 blendImageWithColorMap 함수를 통과 하면서
            # cv2.rectangle 했던 이미지가 없어짐, 블렌딩시 gt와 left 만 섞어서 쓰기때문
            for idx in self.getPointsList:
                rect_start = [int(idx[2]), int(idx[0])]
                rect_end = [int(idx[3]), int(idx[1])]

                thickness = 2    

                # 변수가 계속 살아남게 하여 이미지에 rect가 계속 쌓이는 형식
                self.colormap = cv2.rectangle(
                    self.colormap, rect_start, rect_end, (255, 255, 255), thickness)

                # print(f"rectangle size {rect_start, rect_end}")
                self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
                self.resize_image()
                
        elif event.key() == 77: # m key (ycc)
            
            saveFolderName = os.path.dirname(self.imgPath)
            saveFolderName = os.path.dirname(saveFolderName)
            saveFolderName = os.path.dirname(saveFolderName)
            saveFolderName = os.path.join(saveFolderName, "Coordinate")
            saveImgName = os.path.basename(self.imgPath)
                    
            csvImgName = saveImgName.replace("_leftImg8bit.png", ".csv")
            
            self.getPointsList = []

            with open(os.path.join(saveFolderName, csvImgName), "r", encoding="cp949", newline='') as f :
                data = csv.reader(f)
                for row in data:
                    self.getPointsList.append(row)

            for idx in self.getPointsList:
                AutoLabelButton.pointsRoi_histEq_ycc(self, int(idx[0]), int(idx[1]), int(idx[2]), int(idx[3]))
                
            """
            저장된 좌표의 탐지 결과와 ROI도 같이 보고 싶을때 사용
            """
            # 자동 탐지 라벨링 시 blendImageWithColorMap 함수를 통과 하면서
            # cv2.rectangle 했던 이미지가 없어짐, 블렌딩시 gt와 left 만 섞어서 쓰기때문
            for idx in self.getPointsList:
                rect_start = [int(idx[2]), int(idx[0])]
                rect_end = [int(idx[3]), int(idx[1])]

                thickness = 2    

                # 변수가 계속 살아남게 하여 이미지에 rect가 계속 쌓이는 형식
                self.colormap = cv2.rectangle(
                    self.colormap, rect_start, rect_end, (255, 255, 255), thickness)

                # print(f"rectangle size {rect_start, rect_end}")
                self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
                self.resize_image()
                
                
        elif event.key() == 81: # Q key
            self.labelOpacityCheckBox.setChecked(1-self.labelOpacityCheckBox.isChecked())
            self.labelOpacityOnOff()
            # Brush
            # B_key 한번 press 후 Brush 기능 키고 끄자 

        # Save Image
        elif event.key() == 83 : # S key
            # Save the image
            if self.ControlKey : 
                
                imwrite(self.labelPath, self.label)

                print('Save')
                self.saveFolderName = os.path.dirname(self.imgPath)
                self.saveImgName = os.path.basename(self.imgPath)
                
                print(f"self.saveFolderName : {self.saveFolderName}")
                print(f"self.saveImgName : {self.saveImgName}")
                
                self.situationLabel.setText(self.saveImgName + "을(를) 저장하였습니다.")

                ### 저장과 동시에 roi좌표들을 csv 로 저장 하여 준다.
                # open("file", "mode(default=r)", encoding="cp949", newline='')
                # mode_r = read 전용
                # mode_w = writing 전용, 없는 파일은 만든다. 파일이 존재 해도 새로운 파일로 다시 덮어쓴다.
                # mode_a(appending) = writing 전용, 없는 파일은 만든다. 존재하는 파일을 open시 새로운 행 추가 가능

                # self.csvImgName = self.saveImgName.replace("_leftImg8bit.png", ".csv")

                # open(os.path.join(self.saveFolderName, self.csvImgName), "a", encoding="cp949", newline="")
                
                # getScaledPoint 받아서 csv 파일에 저장 
                # 저장된 csv 파일에서 좌표를 읽어와 autolabel실행 
            
            # Automatic image labeling tool (set_roi)
            elif self.ControlKey == False and self.get_points_roi == False :
                self.set_roi = 1-self.set_roi

                if self.set_roi : 
                    self.roiAutoLabelButton.setChecked(True)

                else : 
                    self.roiAutoLabelButton.setChecked(False)
                
                if self.set_roi_256 :       
                    self.set_roi_256 = 1-self.set_roi_256


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
            # get_points_roi Mode Automatic image labeling tool
            elif self.ControlKey == False and self.get_points_roi == True :
                print("get_points_roi Mode")
                
                # self.get_points_roi 활성화된 상태를 유지 하면서 setRec와 256 넘나들기
                self.get_points_roi_setRec = True
                self.auto_256.setChecked(False)
                self.auto_sr.setChecked(True)
                # self.get_points_roi = False

        # Delete Image
        elif event.key() == 16777223 : # delete key
            print(event.key())
            
            
            if "png" in self.dotSplit_imgPath:
                os.remove(self.imgPath)    
                os.remove(self.labelPath)
                saveFolderName = os.path.dirname(self.imgPath)
                saveImgName = os.path.basename(self.imgPath)
                csvImgName = saveImgName.replace("_leftImg8bit.png", ".csv")
                # csv 파일 존재 시 같이 삭제
                # os.remove(os.path.join(saveFolderName, csvImgName))
            elif "csv" in self.dotSplit_imgPath:
                os.remove(self.imgPath)
            

                
        else :
            print(event.key())
          
    def keyReleaseEvent(self, event):

            # zoom
        if event.key() == Qt.Key_Control:
            self.ControlKey = False
            # QApplication.restoreOverrideCursor()

            # handMove
        elif event.key() == Qt.Key_H:
            self.hKey = False
            QApplication.restoreOverrideCursor()
            
        # brush 기능 중 화면 이동하면 브러시 작동한다
        # mousePress 및 Release def 로 수정
    def scrollAreaMousePress(self, event):

        self.hand_last_point = QPoint(QCursor.pos().x(), QCursor.pos().y())
        print(f"scrollAreaMousePress's pos {self.hand_last_point}")
        
    def scrollAreaMouseMove(self, event):


        delta_y = self.hand_last_point.y() - QCursor.pos().y()
        delta_x = self.hand_last_point.x() -  QCursor.pos().x() 

        print(f"delta_y {delta_y}, delta_x {delta_x}")

        setvalueY = self.scrollArea.verticalScrollBar().value()
        setvalueX = self.scrollArea.horizontalScrollBar().value()
        
        self.scrollArea.verticalScrollBar().setValue(setvalueY + delta_y)
        self.scrollArea.horizontalScrollBar().setValue(setvalueX + delta_x)

        self.hand_last_point = QPoint(QCursor.pos().x(), QCursor.pos().y())

    
    def wheelEventScroll(self, event):
        """
        FIXME: zoom in & out 시 이미지 위치 마우스 포인트에 고정 
        마우스 휠 굴림량을 많이 줄 시 위치가 마우스 포인트가 아닌 다른 곳으로 튄다.
        """
        
        self.mouseWheelAngleDelta = event.angleDelta().y() # -> 1 (up), -1 (down)
        if self.ControlKey:
                
            if self.mouseWheelAngleDelta > 0: 
                self.scale *= 1.1
                width_tobe = self.mainImageViewer.geometry().width() * 1.1
                print(f"self.mainImageViewer.geometry: {self.mainImageViewer.geometry()}")
                height_tobe = self.mainImageViewer.geometry().height() * 1.1
            else : 
                self.scale /= 1.1
                width_tobe = self.mainImageViewer.geometry().width() / 1.1
                height_tobe = self.mainImageViewer.geometry().height() / 1.1

            self.resize_image()

            _width_diff = width_tobe - self.scrollArea.geometry().width()
            _height_diff = height_tobe - self.scrollArea.geometry().height() 

            x_max_img_v = self.mainImageViewer.geometry().width()
            y_max_img_v = self.mainImageViewer.geometry().height()

            set_hor_max = _width_diff + 45 if _width_diff > 0 else 0
            set_ver_max = _height_diff + 45 if _height_diff > 0 else 0

            self.scrollArea.horizontalScrollBar().setRange(0, set_hor_max) 
            self.scrollArea.verticalScrollBar().setRange(0, set_ver_max) 
            
            ver_max = self.scrollArea.verticalScrollBar().maximum()
            hor_max = self.scrollArea.horizontalScrollBar().maximum()
            
            if self.scrollArea.verticalScrollBar().maximum() > 0: 
                setvalueY = self.img_v_y/y_max_img_v*ver_max                
                self.scrollArea.verticalScrollBar().setValue(setvalueY)

            if self.scrollArea.horizontalScrollBar().maximum() > 0: 
                setvalueX = self.img_v_x/x_max_img_v*hor_max
                
                self.scrollArea.horizontalScrollBar().setValue(setvalueX)

        else : 
            scroll_value = self.scrollArea.verticalScrollBar().value()
            self.scrollArea.verticalScrollBar().setValue(scroll_value - self.mouseWheelAngleDelta)

    def resize_image(self):
        size = self.pixmap.size()
        self.scaled_pixmap = self.pixmap.scaled(self.scale * size)
        self.mainImageViewer.setPixmap(self.scaled_pixmap)

    def showHorizontalSliderValue(self):

        self.labelOpacityCheckBox.setChecked(True)

        if abs(self.alpha-(self.lableOpacitySlider.value() / 100)) > 0.03 :
            self.alpha = self.lableOpacitySlider.value() / 100
            self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
            self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
            self.resize_image()    

    def labelOpacityOnOff(self):
        
        if self.labelOpacityCheckBox.isChecked():
            self.alpha = self.lableOpacitySlider.value() / 100
        else : 
            self.alpha = 1 
        
        self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
        self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        self.resize_image()    

    def getListWidgetIndex (self):

        print(f"self.listWidget.currentRow(){self.listWidget.currentRow()}")
        
        self.label_class = self.listWidget.currentRow()
        self.label_segmentation = self.listWidget.currentRow()
        
        if self.use_brush :
            print("Brush")

        elif self.set_roi or self.set_roi_256 or self.get_points_roi or self.get_points_roi_setRec :
            if self.label_segmentation == 1:
                DL_Model.crackModel(self)
            elif self.label_segmentation == 2:
                DL_Model.efflorescenceModel(self)
            elif self.label_segmentation == 3:                  # train_8 : crack, rebar, spalling, effloresence
                DL_Model.rebarExposureModel(self)               # 다른 데이터들 : crack, effloresence, rebar, spalling
            elif self.label_segmentation == 4:
                DL_Model.spallingModel(self)       
                

    def getDlListWidgetIndex(self):
        print(f"dlListidx {self.DL_listWidget.currentRow()}")
        self.DL_class = self.DL_listWidget.currentRow()
        self.model = self.model_list[self.DL_class]
        

if __name__ == "__main__" :

    # Open Chalk window
    app = QApplication(sys.argv)
    myWindow = MainWindow() 
    myWindow.show()

    # Open Stopwatch Window 
    # ClockApp().run()
    # LabelBase.register(name='Roboto',
    #                 fn_regular='./font/Roboto-Thin.ttf',
    #                 fn_bold='./font/Roboto-Medium.ttf')

    sys.exit(app.exec_())
