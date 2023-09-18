import sys

from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from utils.utils import *

sys.path.append("./dnn/mmsegmentation")
from mmseg.apis import init_segmentor


class DL_Model :
    def __init__(self) :
        super().__init__()


        
    def crackModel (self):

        config_file = './dnn/checkpoints/2022.01.06 cgnet general crack 2048/cgnet_2048x2048_60k_CrackAsCityscapes.py'
        checkpoint_file = './dnn/checkpoints/2022.01.06 cgnet general crack 2048/iter_60000.pth'
        self.model = init_segmentor(config_file, 
                                    checkpoint_file, 
                                    device='cuda:0')

        


        model_list = ["crack_model_1"]

        self.DL_listWidget.clear()
        
        for model in model_list:
            print(model)
            self.DL_listWidget.addItem(model)
        
        self.DL_listWidget.setCurrentRow(0)
        self.DL_class = self.DL_listWidget.currentRow()

        self.model_list = [self.model]


    def efflorescenceModel (self):
        
        config_file_efflorescence = './dnn/checkpoints/2022.07.28_cgnet_1024x1024_concrete_efflorescence/cgnet_1024x1024_60k_cityscapes.py'
        checkpoint_file_efflorescence = './dnn/checkpoints/2022.07.28_cgnet_1024x1024_concrete_efflorescence/cgnet_1024x1024_iter_60000.pth'
        self.model = init_segmentor(config_file_efflorescence, 
                                    checkpoint_file_efflorescence,
                                    device='cuda:0')

        config_file_efflorescence_2 = './dnn/checkpoints/2023.05.09_cgnet_2048x2048_60k_efflorescence/cgnet_2048x2048_60k_cityscapes_efflorescence.py'
        checkpoint_file_efflorescence_2 = './dnn/checkpoints/2023.05.09_cgnet_2048x2048_60k_efflorescence/iter_60000.pth'
        self.model_2 = init_segmentor(config_file_efflorescence_2, 
                                    checkpoint_file_efflorescence_2,
                                    device='cuda:0')
        


        model_list = ["efflorescence_model_1", "efflorescence_model_2"]

        self.DL_listWidget.clear()
        
        for model in model_list:
            print(model)
            self.DL_listWidget.addItem(model)

        self.DL_listWidget.setCurrentRow(0)
        print(f"dl_list {self.DL_listWidget.currentRow()}")
        self.DL_class = self.DL_listWidget.currentRow()

        self.model_list = [self.model, self.model_2]

        
        
        
        
        

    def rebarExposureModel (self):

        config_file_rebarexposure = './dnn/checkpoints/cgnet_rebarexposure/cgnet_1024x1024_concrete_rebar_60k_cityscapes.py'
        checkpoint_file_rebarexposure = './dnn/checkpoints/cgnet_rebarexposure/iter_60000.pth'
        self.model = init_segmentor(config_file_rebarexposure, 
                                    checkpoint_file_rebarexposure,
                                    device='cuda:0')

        model_list = ["rebarExposure_model_1"]

        self.DL_listWidget.clear()
        
        for model in model_list:
            print(model)
            self.DL_listWidget.addItem(model)

        self.DL_listWidget.setCurrentRow(0)
        self.DL_class = self.DL_listWidget.currentRow()

        self.model_list = [self.model]

        

    def spallingModel(self):

        config_file_spalling = './dnn/checkpoints/cgnet_spalling/cgnet_1024x1024_concrete_spalling_60k_cityscapes.py'
        checkpoint_file_spalling = './dnn/checkpoints/cgnet_spalling/iter_60000.pth'
        self.model = init_segmentor(config_file_spalling, 
                                    checkpoint_file_spalling,
                                    device='cuda:0')
        
        model_list = ["spalling_model_1"]

        self.DL_listWidget.clear()
        
        for model in model_list:
            print(model)
            self.DL_listWidget.addItem(model)

        self.DL_listWidget.setCurrentRow(0)
        self.DL_class = self.DL_listWidget.currentRow()

        self.model_list = [self.model]

        
        
