#!/usr/bin/env python3
"""
Usage:
    manage.py train [--tub=<tub1,tub2,..tubn>] [--file=<file> ...] (--model=<model>) [--transfer=<model>] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer)] [--continuous] [--aug]

"""
import os
import types


class Config:
    def __init__(self):
        self.CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
        self.DATA_PATH = os.path.join(self.CAR_PATH, 'data')
        self.MODELS_PATH = os.path.join(self.CAR_PATH, 'models')

        #CAMERA
        self.IMAGE_W = 160
        self.IMAGE_H = 120
        self.IMAGE_DEPTH = 3         # default RGB=3, make 1 for mono

        self.BATCH_SIZE = 128                #how many records to use when doing one pass of gradient decent. Use a smaller number if your gpu is running out of memory.
        self.TRAIN_TEST_SPLIT = 0.8          #what percent of records to use for training. the remaining used for validation.
        self.MAX_EPOCHS = 100                #how many times to visit all records of your data
        self.VEBOSE_TRAIN = True             #would you like to see a progress bar with text during training?
        self.USE_EARLY_STOP = True           #would you like to stop the training if we see it's not improving fit?
        self.EARLY_STOP_PATIENCE = 5         #how many epochs to wait before no improvement
        self.MIN_DELTA = .0005               #early stop will want this much loss change before calling it improved.
        self.PRINT_MODEL_SUMMARY = True      #print layers and weights to stdout
        self.OPTIMIZER = None                #adam, sgd, rmsprop, etc.. None accepts default
        self.LEARNING_RATE = 0.001           #only used when OPTIMIZER specified
        self.LEARNING_RATE_DECAY = 0.0       #only used when OPTIMIZER specified
        self.CACHE_IMAGES = True             #keep images in memory. will speed succesive epochs, but crater if not enough mem.

        self.ROI_CROP_TOP = 0                    #the number of rows of pixels to ignore on the top of the image
        self.ROI_CROP_BOTTOM = 0                 #the number of rows of pixels to ignore on the bottom of the image
        self.TARGET_H = self.IMAGE_H - self.ROI_CROP_TOP - self.ROI_CROP_BOTTOM
        self.TARGET_W = self.IMAGE_W
        self.TARGET_D = self.IMAGE_DEPTH
if __name__ == '__main__':
    cfg = Config()
    
    from train import train
    
    tub = None
    model = "./models/mymodel.h5"
    transfer = None
    model_type = None
    continuous = None
    aug = None     

    if tub is not None:
        tub_paths = [os.path.expanduser(n) for n in tub.split(',')]
        dirs.extend( tub_paths )

    train(cfg, None, model, transfer, model_type, continuous, aug)
