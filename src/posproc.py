import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from .config import config

def normalized(in_im):
    """ Rescale im values of shape [W][H] to range 0 -> 255
    """
    return cv.normalize(in_im, None, 0, 255, cv.NORM_MINMAX)    
    
def make_binary(textmap, thresh):
    """
    Parameters
    ----------
    textmap : numpy
    thresh : from 0 to 255

    Returns
    ----------
    textmap : numpy
    """
    textmap = textmap.copy()
    # textmap *= 255.0
    # textmap = np.clip(textmap, 0, 255).astype(np.uint8)
    new_im = np.where(textmap>=thresh,255,0)
    new_im = new_im.astype(np.uint8)
    return new_im

def get_boxes(binary_map, origin_im, debug_name):
    """
    Parameters
    ----------
    binary_map : numpy binary map of values 0, 1

    Returns
    ----------
    ret : boxes
    """
    _, cnts, hierarchy = cv.findContours(binary_map, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    ret = []
    for cnt in cnts:
        # rect = cv.minAreaRect(cnt)
        # box = cv.boxPoints(rect)
        x,y,w,h = cv.boundingRect(cnt)
        u, v = x+w, y+h
        x -= config.BOX_DILATION_RATIO * w
        y -= config.BOX_DILATION_RATIO * h
        u += config.BOX_DILATION_RATIO * w
        v += config.BOX_DILATION_RATIO * h
        box = [[x,y],[u,y],[u,v],[x,v]]
        # print(x,y,u,v)
        box = np.int0(box)

        ret.append(box)
    """
    print("-------")
    origin_im = origin_im.copy()
    for i, box in enumerate(ret):
        x, y, u, v = box[0][0]*2, box[0][1]*2, box[2][0]*2, box[2][1]*2
        print(x,y,u,v)
        cluster = cv.cvtColor(origin_im[y:v, x:u], cv.COLOR_BGR2GRAY)
        
        plt.subplot(1,2,1)
        plt.imshow(cluster)
        
        _, cluster = cv.threshold(cluster,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        plt.subplot(1,2,2)
        plt.imshow(cluster)

        plt.savefig("test_output/{}_cluster_{}.jpg".format(debug_name, str(i)))
    """
    return ret