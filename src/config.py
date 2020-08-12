class Config():
    def __init__(self):
        self.USE_GPU = True

        self.MAX_SIDE = 32 * 32
        self.INPUT_SHAPE = (128,128,1)
            # should be divisible by 32

        self.BOX_DILATION_RATIO = 0.3
        self.SIDE_DIVISOR = 32
        self.MAG_RATIO = 2
        self.TRAIN_LOC = 'data/train'
        self.VAL_LOC = 'data/val'
        self.TEST_LOC = 'data/test'
        
        self.FINE_TUNE = False
        self.CHECKPOINT_PATH = "checkpoints/craft_num_aug10.h5"
        
        self.DEFAULT_GAUSSIAN_MASK_SIDE = 256
        self.BATCH_SIZE = 1
            # At this version, batch_size must be 1 
        self.NUM_EPOCH = 500
        self.LEARNING_RATE = 1
        self.SAVE_GAP = 10


config = Config()
