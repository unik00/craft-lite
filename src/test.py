import matplotlib
#from matplotlib import pyplot as plt
from keras.models import load_model
from . import data_helper
from .config import config
from . import model_keras
from . import posproc

import time
import random
import numpy as np
import cv2 as cv

def validate():
    dir_in = '/home/huynd/Documents/hakaru/notebooks/output/vl'
    test_gen = data_helper.MY_Generator(config.BATCH_SIZE,config.INPUT_SHAPE, dir_in=dir_in, training=False)

    model = dict()
    for i in range(5):
        checkpoint_fold_path = '.'.join(config.CHECKPOINT_PATH.split('.')[:-1]) + "_f{}.h5".format(i+1)
        model[i] = load_model(checkpoint_fold_path)
        print("Loaded model {}.".format(i + 1))

    random.shuffle(test_gen.image_filenames)

    for i,_ in enumerate(test_gen.image_filenames):
        print(test_gen.image_filenames[i])
        start_time = time.time()
        x_batch, _ = test_gen.__getitem__(i)
        pred_im = None
        for j in range(5):
            pred = model[j].predict(x_batch)
            if j == 0:
                pred_im = np.squeeze(pred, axis=0)
            else:
                pred_im += np.squeeze(pred, axis=0)

        pred_im /= 5
        pred_im = posproc.normalized(pred_im)

        #fig = plt.figure()
        #ax1 = fig.add_subplot(221)
        #ax2 = fig.add_subplot(222)
        #ax3 = fig.add_subplot(223)
        #ax4 = fig.add_subplot(224)

        #ax1.title.set_text('Test image')
        #ax2.title.set_text('Test label')    
        #ax3.title.set_text('Predict')    
        #ax4.title.set_text('Boxes')    

        origin = np.squeeze(x_batch, axis=0) * 255
        origin = cv.cvtColor(origin.astype(np.uint8), cv.COLOR_GRAY2RGB)
        
        binary_map = posproc.make_binary(pred_im, 50)
        boxes = posproc.get_boxes(binary_map, origin, str(i))

        for box in boxes:
            box *= 2
            cv.drawContours(origin,[box],0,(0,255,0),2)
            # origin=cv.rectangle(origin, (x,y), (u,v), color=(0,255,0), thickness=1                                                                                                                                                                                                                                                                                                                  )

        #ax1.imshow(val_img)
        #ax2.imshow(val_gt)
        #ax3.imshow(pred_im)
        #ax4.imshow(origin)

        print("inference and processing time for {}x{} image: {}".format(x_batch.shape[1],x_batch.shape[2],time.time()-start_time))
        #plt.show()
        cv.imwrite("test_output/{}.jpg".format(str(i)), origin)
        cv.imwrite("test_output/{}_mask.jpg".format(str(i)), pred_im)

if __name__ == "__main__":
    validate()
    # test()
