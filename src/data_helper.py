import os.path
from pathlib import Path
from math import exp, cos, sin

import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from keras.utils import Sequence
import albumentations as A
import random

from .image_reader import _padding_image
from .config import config

def get_gaussian_grayscale_mask():
    """ 
    Calculate a default isotropic Gaussian Grayscale Mask for a rectangle.
    Probability as a function of distance from the center derived 
    from a gaussian distribution with mean = 0 and stdv = 1.
    
    This code is from: https://github.com/clovaai/CRAFT-pytorch/issues/3
    """
    print("Creating default Gaussian mask...")
    scaled_gaussian = lambda x : exp(-(1/2)*(x**2))

    img_size = config.DEFAULT_GAUSSIAN_MASK_SIDE

    isotropic_grayscale_image = np.zeros((img_size,img_size),np.uint8)

    for i in range(img_size):
      for j in range(img_size):

        """
        find euclidian distance from center of image (img_size/2,img_size/2) 
        and scale it to range of 0 to 2.5 as scaled Gaussian
        returns highest probability for x=0 and approximately
        zero probability for x > 2.5
        """

        distance_from_center = np.linalg.norm(np.array([i-img_size/2,j-img_size/2]))
        distance_from_center = 2.5*distance_from_center/(img_size/2)
        scaled_gaussian_prob = scaled_gaussian(distance_from_center)
        isotropic_grayscale_image[i,j] = np.clip(scaled_gaussian_prob*255,0,255)

    # Convert Grayscale to HeatMap Using Opencv
    # isotropicGaussianHeatmapImage = cv2.applyColorMap(isotropic_grayscale_image, 
                                                      # cv2.COLORMAP_JET)

    # plt.imshow(isotropicGaussianHeatmapImage)
    # plt.imshow(isotropic_grayscale_image)
    # plt.show()
    print("Finished generating default Gaussian mask.")
    return isotropic_grayscale_image

intial_mask = get_gaussian_grayscale_mask()

def get_normal_char_mask(W, H):
    """
    Parameters
    ----------
    W , H : positive integer
        width and height of the rectangle
    """
    ret = cv2.resize(intial_mask, (W, H),interpolation=cv2.INTER_NEAREST)    
    # plt.imshow(ret)
    # plt.show()
    return ret

def get_perspective_transformed(image, pts):
    """
    Parameters
    ----------
    image : numpy 2D array
        image Bird-eyed image to be transformed into pts
    pts : numpy array of shape [None, 2]
        Polygon perspective 
    
    Returns
    ----------
    mask : numpy 2D array
    (x, y, u, v): pair of float
        Coordinate of the mask on the image where polygon pts is located
    """

    min_x, min_y = np.min(pts[:, 0]).astype(np.int32), np.min(pts[:, 1]).astype(np.int32)
    max_x, max_y = np.max(pts[:, 0]).astype(np.int32), np.max(pts[:, 1]).astype(np.int32)
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    
    dst = np.array([
        [0, 0],
        [image.shape[1] - 1, 0],
        [image.shape[1] - 1, image.shape[0] - 1],
        [0, image.shape[0] - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(dst, pts)
    warped = cv2.warpPerspective(image, M, (max_x, max_y),flags=cv2.INTER_NEAREST)
    warped = warped[min_y:,min_x:]
    return warped, (min_x, min_y, max_x, max_y)

def padded_to_multiple(in_im):
    """ Resize image sides to multiples of SIDE_DIVISOR as required by the network.
    Parameters
    ----------
    in_im : numpy array of shape [original_h][original_w]
        Gray scale image, pixel in range [0, 1].

    Returns
    ----------
    out_im : numpy array of shape [new_h][new_w] 
        Where new_h >= original_h, new_w >= original w
        and  new_h, new_w are divisible by SIDE_DIVISOR.
    """
    im = in_im.copy()
    
    H, W = im.shape
    new_H, new_W = H, W

    if H % config.SIDE_DIVISOR != 0:
        new_H = H + config.SIDE_DIVISOR - H % config.SIDE_DIVISOR
    
    if W % config.SIDE_DIVISOR != 0:
        new_W = W + config.SIDE_DIVISOR - W % config.SIDE_DIVISOR

    canvas = np.zeros((new_H, new_W), dtype=np.float64)
    canvas[0:H, 0:W] = im
    out_im = canvas
    return out_im

class MY_Generator(Sequence):
    def __init__(self, dir_in, batch_size, input_shape, training=False):
        self.image_filenames = self.get_images(dir_in)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.training = training
        
    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def get_images(self, dir_in):
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
        

        filtered_img_files = []

        for img_file in all_img_files:
            anno_path = '.'.join(str(img_file).split('.')[:-1]) + ".xml"
            if os.path.exists(anno_path):
                filtered_img_files.append(str(img_file))
            else:
                print("CANNOT FIND ANNOTATION FOR {}, SKIPPED".format(str(img_file)))
        
        # print(filtered_img_files)

        return filtered_img_files

    def _augmented(self, image, mask):
        # always train on crops
        max_w = min(image.shape[0], image.shape[1])

        light = A.Compose([
            A.RandomSizedCrop((320, max_w), 512, 512),
            A.ShiftScaleRotate(),
            A.Blur(),
            # A.GaussNoise(),
            # A.ElasticTransform(border_mode=cv2.BORDER_REPLICATE),
            A.ElasticTransform(),
            A.MaskDropout((10,15), p=1),
            A.Cutout(p=1)
        ],p=1)
        augmented = light(image=image,mask=mask)
        return augmented['image'], augmented['mask']

    def __getitem__(self, idx):
        x_batch=[]
        y_batch =[]

        img_paths = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        for img_path in img_paths:
            img_path = str(img_path)
            #print(img_path)
            anno_path = '.'.join(img_path.split('.')[:-1]) + ".xml"
            #anno_path = anno_path.replace('/imgs','/anns')

            original_im = cv2.imread(img_path)
            img = cv2.imread(img_path)

            assert self.batch_size == 1

            # magnify the image, not yet actually resize 
            self.input_shape = (int(img.shape[0] * config.MAG_RATIO), int(img.shape[1] * config.MAG_RATIO), 1)
            
            # resize the image to config.MAX_SIDE if it's too large
            if max(self.input_shape[0], self.input_shape[1]) > config.MAX_SIDE:
                ratio = config.MAX_SIDE / max(self.input_shape[0], self.input_shape[1])
                self.input_shape = (int(self.input_shape[0] * ratio), int(self.input_shape[1] * ratio),self.input_shape[-1])
            
            # make the image size multiple of 32, not yet actually resize        
            h = self.input_shape[0]
            w = self.input_shape[1]
            self.input_shape = (max(32,(h//32)*32),max(64,(w//32)*32),self.input_shape[-1])
            
            # convert the image to gray and normalize it
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) 
            img = img.astype(np.float64)
            img /= 255.0
            mask = np.zeros(img.shape)

            if os.path.exists(anno_path):
                has_anno = True
                tree = ET.parse(anno_path)
                root = tree.getroot()

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
                     #   print(box, cx,cy,w,h)
                        w += 2 * config.BOX_DILATION_RATIO * w
                        h += 2 * config.BOX_DILATION_RATIO * h
                        w /= 2
                        h /= 2

                        x = cx - w/2
                        y = cy - h/2
                        u = cx + w/2
                        v = cy + h/2

                        do_rotate=lambda x, y, theta: [x*cos(angle) - y*sin(angle), y*cos(angle) + x*sin(angle)]
                        pts = np.float32([[x,y],[u,y],[u,v],[x,v]])
                        
                        for i, _ in enumerate(pts):
                            pts[i] = do_rotate(pts[i][0]-cx,pts[i][1]-cy,angle)
                            pts[i][0] += cx
                            pts[i][1] += cy                            
                          
                        iggm, rec = get_perspective_transformed(intial_mask, pts)

                        mask[rec[1]:(rec[3]),rec[0]:(rec[2])] += iggm
                        
                    else:
                        box = (int(member[5][0].text),
                                int(member[5][1].text),
                                int(member[5][2].text),
                                int(member[5][3].text)
                                )
                        x, y, u, v = box
                        w = u - x
                        h = v - y

                        x -= config.BOX_DILATION_RATIO * w
                        y -= config.BOX_DILATION_RATIO * h
                        u += config.BOX_DILATION_RATIO * w
                        v += config.BOX_DILATION_RATIO * h

                        x = int(x)
                        y = int(y)
                        u = int(u)
                        v = int(v)
                        #img = cv2.rectangle(img,(x, y), (u, v),color=1.,thickness=1)
                        iggm = get_normal_char_mask((u-x+1), (v-y+1))
                        for i in range(u-x+1):
                            for j in range(v-y+1):
                                jj, ii = y+j, x+i
                                jj = min(jj, img.shape[0] - 1)
                                ii = min(ii, img.shape[1] - 1)
                                mask[jj][ii] += iggm[j][i]
            else:
                has_anno = False
           
            img = cv2.resize(img, (self.input_shape[1],self.input_shape[0]))
            mask = cv2.resize(mask, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_NEAREST)

            if self.training:
                img, mask = self._augmented(img, mask)
                
                plt.subplot(1,2,1)
                plt.imshow(img)
                plt.subplot(1,2,2)
                plt.imshow(mask)
                plt.show()
                
            if has_anno:
                assert img.shape[0] % 2 == 0 and img.shape[1] % 2 == 0
                mask = cv2.resize(mask, (img.shape[1]//2,img.shape[0]//2), interpolation=cv2.INTER_NEAREST)
            else:
                print("{} DOESNT HAVE ANNOTATION".format(img_path))
                mask = None
            
            img = np.expand_dims(img, -1)
            mask = np.expand_dims(mask, -1)
            x_batch.append(img)
            y_batch.append(mask)
            
        return np.array(x_batch),np.array(y_batch)


if __name__ == "__main__":
    train_generator = MY_Generator(config.TRAIN_LOC,config.BATCH_SIZE,config.INPUT_SHAPE,training=True)
    for _ in range(10):
        train_generator.__getitem__(300)
    pass

