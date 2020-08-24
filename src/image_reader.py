import cv2
import os, os
import cv2
import glob
import numpy as np
import unicodedata
import random
from pathlib import Path

def get_images(dir_in, filter_label=True):
    ''' Get the images in a directory RECURSIVELY, ignore images without .xml extentions
    Parameters
    ----------
    dir_in : str
        Root directory to be search from

    Returns
    ----------
    filtered_img_files : list of str
    '''

    img_exts = ['jpg', 'jpeg', 'png', 'webp', 'gif']
    all_img_files = []

    # get all img files    
    for img_ext in img_exts:
        all_img_files += list(Path(dir_in).rglob('*.{}'.format(img_ext.lower())))
        all_img_files += list(Path(dir_in).rglob('*.{}'.format(img_ext.upper())))
    
    if not filter_label:
        filtered_img_files = all_img_files
    else:
        filtered_img_files = []
        for img_file in all_img_files:
            anno_path = '.'.join(str(img_file).split('.')[:-1]) + ".xml"
            if os.path.exists(anno_path):
                filtered_img_files.append(str(img_file))
            else:
                print("CANNOT FIND ANNOTATION FOR {}, SKIPPED".format(str(img_file)))
    
        # print(filtered_img_files)

    return filtered_img_files

def get_restore(image,shape,mask=None):
    net_h, net_w,net_c = shape if mask is None else mask.shape

    if image is None:
        if net_c == 1:
            new_image = np.zeros((net_h, net_w, net_c))
        else:
            new_image = np.ones((net_h, net_w, net_c))
    else:
        new_h, new_w = image.shape[:2]
        # determine the new size of the image
        if net_w < new_w or net_h < new_h:
            if (float(net_w)/new_w) < (float(net_h)/new_h):
                new_h = (new_h * net_w)//new_w
                new_w = net_w
            else:
                new_w = (new_w * net_h)//new_h
                new_h = net_h

        # resize the image to the new size
        restore_rat = (image.shape[0] / new_h, image.shape[1] / new_w) 
    return restore_rat

def _padding_image(image,shape,is_center = False,mask=None):
    net_h, net_w,net_c = shape if mask is None else mask.shape
    if image is None:
        if net_c == 1:
            new_image = np.zeros((net_h, net_w, net_c))
        else:
            new_image = np.ones((net_h, net_w, net_c))
    else:
        new_h, new_w = image.shape[:2]
        # determine the new size of the image
        if net_w < new_w or net_h < new_h:
            if (float(net_w)/new_w) < (float(net_h)/new_h):
                new_h = (new_h * net_w)//new_w
                new_w = net_w
            else:
                new_w = (new_w * net_h)//new_h
                new_h = net_h

        # resize the image to the new size
        resized = cv2.resize(image, (new_w, new_h))
        if net_c == 1:
            resized = np.expand_dims(resized, -1)
            # embed the image into the standard letter box
            new_image = np.zeros((net_h, net_w, net_c))
        else:
            # embed the image into the standard letter box
            if mask is None:
                new_image = np.ones((net_h, net_w, net_c))
            else:
                new_image =mask
        if is_center:  
            new_image[(net_h-new_h)//2:(net_h+new_h)//2, (net_w-new_w)//2:(net_w+new_w)//2, :] = resized
        else:
            new_image[:new_h,:new_w, :] = resized

    # new_image = new_image.astype(np.float32)
    # new_image = new_image/255.
    return new_image


