import random
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

#from matplotlib import pyplot as plt
from . import data_helper
from .config import config
from . import model_keras
from .image_reader import get_images


def one_fold():
    train_generator = data_helper.MY_Generator(config.BATCH_SIZE,config.INPUT_SHAPE, dir_in=config.TRAIN_LOC, training=True)
    val_generator = data_helper.MY_Generator(config.BATCH_SIZE,config.INPUT_SHAPE, dir_in=config.VAL_LOC, training=False)

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

def five_fold_cv(dir_in):
    random.seed(11)
    all_imgs = get_images(dir_in)
    random.shuffle(all_imgs)
    fraction = int(0.2 * len(all_imgs))
    
    result = dict()

    for i in range(0,5):
        val_generator = data_helper.MY_Generator(config.BATCH_SIZE,config.INPUT_SHAPE, dir_in=None, training=False)

        val_left = i*fraction
        val_right = (i+1)*fraction
        val_generator.image_filenames = all_imgs[val_left:val_right]
        val_generator.image_filenames = all_imgs[i*fraction : (i+1)*fraction]

        train_generator = data_helper.MY_Generator(config.BATCH_SIZE,config.INPUT_SHAPE, dir_in=None, training=True)
        train_generator.image_filenames = all_imgs[:val_left] + all_imgs[val_right:]

        model = model_keras.craft()
        #model.load_weights(config.CHECKPOINT_PATH, by_name=True, skip_mismatch=True)

        checkpoint_fold_path = '.'.join(config.CHECKPOINT_PATH.split('.')[:-1]) + "_f{}.h5".format(i+1)

        # Save the model according to the conditions
        checkpoint = ModelCheckpoint(checkpoint_fold_path, monitor='loss', verbose=1, save_best_only=True,
                                        save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor='loss', min_delta=0, patience=max(10, config.NUM_EPOCH // 5), verbose=1, mode='auto')

        print("-------FOLD {}----------".format(i + 1))
        print("Length val_generator for fold {}: {}".format(i + 1, len(val_generator.image_filenames)))
        print("Length train_generator for fold {}: {}".format(i + 1, len(train_generator.image_filenames)))
        # Train the model
        history = model.fit_generator(
            generator            = train_generator,
            epochs               = config.NUM_EPOCH,
            validation_data      = val_generator,
            callbacks            = [checkpoint, early],
            workers              = 6,
            max_queue_size       = 12
            )
        print("Result for fold {}: ".format(i + 1))
        print(history.history)
        if i == 0:
            result['val_loss'] = 0
            result['loss'] = 0
        result['val_loss'] += history.history['val_loss'][-1]
        result['loss'] += history.history['loss'][-1]
        K.clear_session()
        del model
    
    result['val_loss'] /= 5
    result['loss'] /= 5
    print("CV result:\n\t Train loss: {}, Validation loss: {}".format(result['loss'], result['val_loss']))
    
if __name__ == "__main__":
    # one_fold()
    five_fold_cv(config.TRAIN_LOC)
