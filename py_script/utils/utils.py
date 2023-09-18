import os 
import sys

import cv2

import numpy as np 

from PyQt5.QtGui import QImage
from PyQt5.QtCore import QPoint

sys.path.append("./dnn/mmsegmentation")
from mmseg.apis import inference_segmentor


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

def histEqualization_gr (img):
    ## [Convert to grayscale(binary)]
    src_gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ## [Convert to grayscale(binary)]

    ## [Apply Histogram Equalization]
    dst_gr = cv2.equalizeHist(src_gr)
    ## [Apply Histogram Equalization]

    ## [Convert to bgrimage]
    dst_gr_bgr = cv2.cvtColor(dst_gr, cv2.COLOR_GRAY2BGR)
    ## [Convert to bgrimage]
    
    return dst_gr_bgr

def histEqualization_hsv (img):
    # hsv 컬러 형태로 변형합니다.
    src_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # h(hue: 색상), s(saturation: 채도), v(value: 명도(색의 밝기))로 컬러 영상을 분리 합니다. 
    h, s, v = cv2.split(src_hsv)
    # v(value: 명도)값을 히스토그램 평활화를 합니다.
    dst_hsv_v = cv2.equalizeHist(v)
    # h,s,equalizedV를 합쳐서 새로운 hsv 이미지를 만듭니다.
    dst_hsv_merged = cv2.merge([h,s,dst_hsv_v])
    # 마지막으로 hsv2를 다시 BGR 형태로 변경합니다.
    dst_hsv_merged_bgr = cv2.cvtColor(dst_hsv_merged, cv2.COLOR_HSV2BGR)

    return dst_hsv_merged_bgr

def histEqualization_ycc (img):
    # YCrCb 컬러 형태로 변환합니다.
    src_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # y(휘도(색의 밝기)), Cr(색차 성분), Cb(색차 성분)로 컬러 영상을 분리 합니다.
    y, Cr, Cb = cv2.split(src_ycc)
    # y값을 히스토그램 평활화를 합니다.
    dst_ycc_y = cv2.equalizeHist(y)
    # equalizedY, Cr, Cb를 합쳐서 새로운 yCrCb 이미지를 만듭니다.
    dst_ycc_merged = cv2.merge([dst_ycc_y, Cr, Cb])
    # 마지막으로 yCrCb2를 다시 BGR 형태로 변경합니다.
    dst_ycc_merged_bgr = cv2.cvtColor(dst_ycc_merged, cv2.COLOR_YCrCb2BGR)

    return dst_ycc_merged_bgr


def pointsRoi(model, src, label, label_segmentation,y_start, y_end, x_start, x_end):
        
    crop_img = src[y_start: y_end, x_start: x_end, :]
    result = inference_segmentor(model, crop_img)
    
    idx = np.argwhere(result[0] == 1)
    y_idx, x_idx = idx[:, 0], idx[:, 1]
    x_idx = x_idx + x_start
    y_idx = y_idx + y_start

    label[y_idx, x_idx] = label_segmentation

    return label

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

