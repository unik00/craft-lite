#from matplotlib import pyplot as plt
from . import data_helper
from .config import config
from . import model_keras

import random
import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping

def filtered_data(in_datas):
        # filter images without annotations
    datas = in_datas.copy()
    for i in range(len(datas)):
        if not datas[i][3]:
            print("Cannot find annotation for {}, skip the image!".format(datas[i][4]))

    valid_data_indexes = [i for i in range(len(datas)) if datas[i][3]]
    datas = [datas[i] for i in valid_data_indexes]
    return datas

def main():
    print("Loading training data...")

    train_generator = data_helper.MY_Generator(config.TRAIN_LOC,config.BATCH_SIZE,config.INPUT_SHAPE)
    val_generator = data_helper.MY_Generator(config.VAL_LOC,config.BATCH_SIZE,config.INPUT_SHAPE)
    print("Finished loading training data.")
    model = model_keras.craft()
#    model.load_weights(config.CHECKPOINT_PATH, by_name=True, skip_mismatch=True)

    # Save the model according to the conditions
    checkpoint = ModelCheckpoint(config.CHECKPOINT_PATH, monitor='loss', verbose=1, save_best_only=True,
                                    save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='loss', min_delta=0, patience=5000, verbose=1, mode='auto')

    # Train the model
    history = model.fit_generator(
        generator            = train_generator,
        epochs               = 5000,
        validation_data      = val_generator,
        callbacks            = [checkpoint, early],
        workers              = 6,
        max_queue_size       = 12
        )


if __name__ == "__main__":
    main()
