import time
import random
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from math import sqrt
from matplotlib import pyplot as plt
from keras.models import load_model

from . import data_helper
from .config import config
from . import model_keras
from . import posproc
from src.image_reader import get_restore

'''
Percentage of Detected Joints - PDJ: 
    A detected joint is considered correct if the distance between the predicted 
    and the true joint is within a certain fraction of the torso diameter. 
    PDJ@0.2 = distance between predicted and true joint < 0.2 * torso diameter.
    
    https://nanonets.com/blog/human-pose-estimation-2d-guide/

mean Percentage of Detected Joints - mPDJ:
    Mean of PDJ of all cases  
'''

def get_pts(anno_path):
    '''
    Parameters:
    ----------
    anno_path : string 
    '''
    tree = ET.parse(anno_path)
    root = tree.getroot()
    ret = []
    for member in root.findall('object'):
        if member[1].text == "bar":
            continue
        if member[0].text == "robndbox":
        # is rotated bounding box format
            box = (float(member[5][0].text),
                    float(member[5][1].text),
                    float(member[5][2].text),
                    float(member[5][3].text),
                    float(member[5][4].text)
                   )
            cx,cy,w,h,angle = box
            ret.append((cx, cy))
        else:
            # is normal bounding box format
            box = (int(member[5][0].text),
                    int(member[5][1].text),
                    int(member[5][2].text),
                    int(member[5][3].text)
                    )
            x, y, u, v = box
            x = int(x)
            y = int(y)
            u = int(u)
            v = int(v)
            cx = x + (u - x) / 2
            cy = y + (v - y) / 2
            ret.append((cx, cy))
    return ret   


def squared_euclidean_dist(pt1, pt2):
    sqr = lambda x : x * x
    return sqr(pt1[0]-pt2[0]) + sqr(pt1[1]-pt2[1])

def get_min_squared_dist(pts):
    assert pts is not None
    if len(pts) < 2:
        return 0
    ret = squared_euclidean_dist(pts[0], pts[1])
    for i in range(len(pts) - 1):
        for j in range(i + 1, len(pts)):
            ret = min(ret, squared_euclidean_dist(pts[i], pts[j]))
    return ret


def mPDJ():
    test_gen = data_helper.MY_Generator(config.VAL_LOC,1,config.INPUT_SHAPE)
    model = load_model(config.CHECKPOINT_PATH)
    
    m_f1 = 0
    m_recall = 0
    m_precision = 0

    for im_num,_ in enumerate(test_gen.image_filenames):
        img_path = test_gen.image_filenames[im_num]
        print(img_path)
        anno_path = '.'.join(img_path.split('.')[:-1]) + ".xml"
        im = cv2.imread(img_path)
        gt_pts = get_pts(anno_path)
        alpha = 0.4
        thresh = alpha * sqrt(get_min_squared_dist(gt_pts))
        for pt in gt_pts:
            im = cv2.circle(im, (int(pt[0]),int(pt[1])), 0, color=(0,255,0), thickness=5)
        
        # plt.imshow(im)
        # plt.show()

        x_batch, _ = test_gen.__getitem__(im_num)
        pred = model.predict(x_batch)
        pred_im = np.squeeze(pred, axis=0)
        pred_im = posproc.normalized(pred_im)
        binary_map = posproc.make_binary(pred_im, 50)
        pred_pts = posproc.get_points(binary_map)
        origin = np.squeeze(x_batch, axis=0) * 255
        origin = cv2.cvtColor(origin.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        restore_rat = get_restore(im, origin.shape)

        for i, _ in enumerate(pred_pts):
            # pred_pts[i] = pred_pts[i].astype(np.float32)
            for j in range(0,2):
                pred_pts[i][j] *= restore_rat[1 - j % 2] * 2


        for pt in pred_pts:
            # print(pt)
            im = cv2.circle(im, (int(pt[0]),int(pt[1])), int(thresh*2), color=(255,0,0), thickness=2)
            
        cv2.imwrite("metric_debug/{}.jpg".format(str(im_num)), im)
        # plt.imshow(im)
        # plt.show()
        
        cnt = 0
        fp = 0
        fn = 0
        tp = 0
        for gt_pt in gt_pts:
            found = False
            for pred_pt in pred_pts:
                if sqrt(squared_euclidean_dist(gt_pt, pred_pt)) <= thresh:
                    found = True
                    break
            if not found:
                fn += 1

        used = dict()
        for pred_pt in pred_pts:
            found = False
            for i, gt_pt in enumerate(gt_pts):
                if (i not in used) and sqrt(squared_euclidean_dist(gt_pt, pred_pt)) <= thresh:
                    used[i] = True
                    tp += 1
                    found = True
                    break
            if not found:
                fp += 1

        if tp + fp == 0:
        	precision = 0
        else:
        	precision = tp / (tp + fp)
        
        if tp + fn == 0:
        	recall = 0
        else:	
        	recall = tp / (tp + fn)
        
        if tp == 0:
            f1 = 0        
        else:
            f1 = 2 * (precision*recall) / (precision + recall)
        print("Evaluation of {}.jpg".format(str(im_num)))
        print("Precision: {:.2f} - Recall: {:.2f} - F1 score: {:.2f}".format(precision, recall, f1))
        print("(TP: {} - FP: {} - FN: {} )".format(tp, fp, fn))
        m_f1 += f1
        m_precision += precision
        m_recall += recall

    m_f1 /= len(test_gen.image_filenames)
    m_precision /= len(test_gen.image_filenames)
    m_recall /= len(test_gen.image_filenames)

    print("mF1: {} - mP: {} - mR: {}".format(m_f1, m_precision, m_recall))

if __name__ == "__main__":
    mPDJ()
    pass