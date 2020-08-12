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
    test_gen = data_helper.MY_Generator('/home/huynd/Documents/hakaru/notebooks/output/ОlКp_1Тi_РФВаВш',config.BATCH_SIZE,config.INPUT_SHAPE)
    model = load_model(config.CHECKPOINT_PATH)
    model.summary()
    
    random.shuffle(test_gen.image_filenames)

    for i,_ in enumerate(test_gen.image_filenames):
        start_time = time.time()
        x_batch, _ = test_gen.__getitem__(i)
        pred = model.predict(x_batch)
        pred_im = np.squeeze(pred, axis=0)
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
