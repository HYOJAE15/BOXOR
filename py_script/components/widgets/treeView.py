
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from utils_boxor.utils import *

import csv

class TreeView() :
    def __init__(self) :
        super().__init__()

    
    ######################## 
    ### Folder Tree View ###
    ########################

    
    def treeViewImage(self, index) :

        try : 
            
            indexItem = self.treeModel.index(index.row(), 0, index.parent())
            
            self.imgPath = self.treeModel.filePath(indexItem)
            self.fileNameLabel.setText(self.imgPath)
            self.dotSplit_imgPath = self.imgPath.split('.')
            
            if 'png' in self.dotSplit_imgPath :

                self.labelPath = self.imgPath.replace('/leftImg8bit/', '/gtFine/')
                self.labelPath = self.labelPath.replace( '_leftImg8bit.png', '_gtFine_labelIds.png')

                self.img = imread(self.imgPath)
                self.src = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
                self.label = imread(self.labelPath)
                
                self.layers = createLayersFromLabel(self.label, len(self.label_palette))
                self.colormap = blendImageWithColorMap(self.img, self.label, self.label_palette, self.alpha)
                
                self.pixmap = QPixmap(cvtArrayToQImage(self.colormap))
                self.scale = self.scrollArea.height() / self.pixmap.height()

                self.resize_image()

                self.situationLabel.clear()
                self.saveImgName = None
                self.brushMemory = None

            elif "csv" in self.dotSplit_imgPath:
                f = open(self.imgPath, "r", encoding="cp949", newline='')
                data = csv.reader(f)
                self.getPointsList = []

                for row in data:
                    
                    self.getPointsList.append(row)



            else :
                pass
              
        
        except: 
            print("Error Occured")
    

    def askSave(self) :
        
        print("askSave")
        
