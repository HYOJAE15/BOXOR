import os 
import sys

import cv2

import numpy as np 

from PyQt5.QtGui import QImage
from PyQt5.QtCore import QPoint

def imread(imgPath):
    img = cv2.imdecode(np.fromfile(imgPath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img.ndim == 3 : 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imwrite(path, img): 
    _, ext = os.path.splitext(path)
    _, label_to_file = cv2.imencode(ext, img)
    label_to_file.tofile(path)
    

def createLayersFromLabel(label, num_class):

    layers = []

    for idx in range(num_class):
        print(f"index {idx}")
        layers.append(label == idx)
        
    return layers


def getScaledPoint(event, scale):
    """Get scaled point coordinate 
    Args: 
        event (PyQt5 event)
        scale (float)

    Returns:
        x, y (PyQt5 Qpoint)
    """

    scaled_event_pos = QPoint(round(event.pos().x() / scale), round(event.pos().y() / scale))
    x, y = scaled_event_pos.x(), scaled_event_pos.y()

    return x, y 

def resource_path(relative_path): 
    """ 
    Get absolute path to resource, works for dev and for PyInstaller 

    Args :
        relative_path (str)
    
    Return 
        abs_path (str)
    """ 
    
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__))) 
    abs_path = os.path.join(base_path, relative_path)
    
    return abs_path

def cvtArrayToQImage(array):

    if len(array.shape) == 3 : 
        h, w, _ = array.shape
    else :
        raise 
    
    return QImage(array.data, w, h, 3 * w, QImage.Format_RGB888)

def blendImageWithColorMap(image, label, palette, alpha):
    """ blend image with color map 
    Args: 
        image (3d np.array): RGB image
        label (2d np.array): 1 channel gray-scale image
        pallete (2d np.array) 
        alpha (float)

    Returns: 
        color_map (3d np.array): RGB image
    """

    color_map = np.zeros_like(image)
        
    for idx, color in enumerate(palette) : 
        
        if idx == 0 :
            color_map[label == idx, :] = image[label == idx, :] * 1
        else :
            color_map[label == idx, :] = image[label == idx, :] * alpha + color * (1-alpha)

    return color_map



def points_between(x1, y1, x2, y2):
    """
    coordinate between two points
    """

    d0 = x2 - x1
    d1 = y2 - y1
    
    count = max(abs(d1)+1, abs(d0)+1)

    if d0 == 0:
        return (
            np.full(count, x1),
            np.round(np.linspace(y1, y2, count)).astype(np.int32)
        )

    if d1 == 0:
        return (
            np.round(np.linspace(x1, x2, count)).astype(np.int32),
            np.full(count, y1),  
        )

    return (
        np.round(np.linspace(x1, x2, count)).astype(np.int32),
        np.round(np.linspace(y1, y2, count)).astype(np.int32)
    )

def make_cityscapes_format_points (image, save_dir, damage_type) :
    temp_img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    gt = np.zeros((temp_img.shape[0], temp_img.shape[1]), dtype=np.uint8)

    
    
    save_dir_img = os.path.join(save_dir, 'leftImg8bit', damage_type)
    save_dir_gt = os.path.join(save_dir, 'gtFine', damage_type)

    os.makedirs(save_dir_img, exist_ok=True)
    os.makedirs(save_dir_gt, exist_ok=True)

    img_filename = os.path.basename(image)
    # img_filename = img_filename.replace('.png', '_leftImg8bit.png')
    
    gt_filename = img_filename.replace('_leftImg8bit.png', '_gtFine_labelIds.png')

    is_success, org_img = cv2.imencode(".png", temp_img)
    org_img.tofile(os.path.join(save_dir_img, img_filename))

    is_success, gt_img = cv2.imencode(".png", gt)
    gt_img.tofile(os.path.join(save_dir_gt, gt_filename))
    gt_path = os.path.join(save_dir_gt, gt_filename) 
    
    return gt_path

def make_cityscapes_format_imagetype (image, save_dir, image_type) :
    temp_img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    gt = np.zeros((temp_img.shape[0], temp_img.shape[1]), dtype=np.uint8)

    
    
    save_dir_img = os.path.join(save_dir, 'leftImg8bit')
    save_dir_gt = os.path.join(save_dir, 'gtFine')

    os.makedirs(save_dir_img, exist_ok=True)
    os.makedirs(save_dir_gt, exist_ok=True)

    img_filename = os.path.basename(image)
    img_filename = img_filename.replace(f'.{image_type}', '_leftImg8bit.png')
    
    gt_filename = img_filename.replace('_leftImg8bit.png', '_gtFine_labelIds.png')

    is_success, org_img = cv2.imencode(".png", temp_img)
    org_img.tofile(os.path.join(save_dir_img, img_filename))

    is_success, gt_img = cv2.imencode(".png", gt)
    gt_img.tofile(os.path.join(save_dir_gt, gt_filename))
    gt_path = os.path.join(save_dir_gt, gt_filename) 
    
    return gt_path

