'''
utils.py

Functions that don't fit anywhere else.

'''
from io import BytesIO
import os
import glob
import socket
import zipfile
import sys
import itertools
import subprocess
import math
import random
import time

from PIL import Image
import numpy as np

'''
IMAGES
'''

def scale(im, size=128):
    '''
    accepts: PIL image, size of square sides
    returns: PIL image scaled so sides lenght = size 
    '''
    size = (size,size)
    im.thumbnail(size, Image.ANTIALIAS)
    return im


def img_to_binary(img, format='jpeg'):
    '''
    accepts: PIL image
    returns: binary stream (used to save to database)
    '''
    f = BytesIO()
    try:
        img.save(f, format=format)
    except Exception as e:
        raise e
    return f.getvalue()


def arr_to_binary(arr):
    '''
    accepts: numpy array with shape (Hight, Width, Channels)
    returns: binary stream (used to save to database)
    '''
    img = arr_to_img(arr)
    return img_to_binary(img)


def arr_to_img(arr):
    '''
    accepts: numpy array with shape (Height, Width, Channels)
    returns: binary stream (used to save to database)
    '''
    arr = np.uint8(arr)
    img = Image.fromarray(arr)
    return img

def img_to_arr(img):
    '''
    accepts: numpy array with shape (Height, Width, Channels)
    returns: binary stream (used to save to database)
    '''
    return np.array(img)


def binary_to_img(binary):
    '''
    accepts: binary file object from BytesIO
    returns: PIL image
    '''
    if binary is None or len(binary) == 0:
        return None

    img = BytesIO(binary)
    try:
        img = Image.open(img)
        return img
    except:
        return None


def norm_img(img):
    return (img - img.mean() / np.std(img))/255.0


def rgb2gray(rgb):
    '''
    take a numpy rgb image return a new single channel image converted to greyscale
    '''
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def img_crop(img_arr, top, bottom):
    
    if bottom is 0:
        end = img_arr.shape[0]
    else:
        end = -bottom
    return img_arr[top:end, ...]

def normalize_and_crop(img_arr, cfg):
    img_arr = img_arr.astype(np.float32) / 255.0
    if cfg.ROI_CROP_TOP or cfg.ROI_CROP_BOTTOM:
        img_arr = img_crop(img_arr, cfg.ROI_CROP_TOP, cfg.ROI_CROP_BOTTOM)
        if len(img_arr.shape) == 2:
            img_arrH = img_arr.shape[0]
            img_arrW = img_arr.shape[1]
            img_arr = img_arr.reshape(img_arrH, img_arrW, 1)
    return img_arr


def load_scaled_image_arr(filename, cfg):
    '''
    load an image from the filename, and use the cfg to resize if needed
    also apply cropping and normalize
    '''
    try:
        img = Image.open(filename)
        if img.height != cfg.IMAGE_H or img.width != cfg.IMAGE_W:
            img = img.resize((cfg.IMAGE_W, cfg.IMAGE_H))
        img_arr = np.array(img)
        img_arr = normalize_and_crop(img_arr, cfg)
        croppedImgH = img_arr.shape[0]
        croppedImgW = img_arr.shape[1]
        if img_arr.shape[2] == 3 and cfg.IMAGE_DEPTH == 1:
            img_arr = rgb2gray(img_arr).reshape(croppedImgH, croppedImgW, 1)
    except Exception as e:
        print(e)
        print('failed to load image:', filename)
        img_arr = None
    return img_arr

        

'''
FILES
'''


def most_recent_file(dir_path, ext=''):
    '''
    return the most recent file given a directory path and extension
    '''
    query = dir_path + '/*' + ext
    newest = min(glob.iglob(query), key=os.path.getctime)
    return newest


def make_dir(path):
    real_path = os.path.expanduser(path)
    if not os.path.exists(real_path):
        os.makedirs(real_path)
    return real_path


def zip_dir(dir_path, zip_path):
    """ 
    Create and save a zipfile of a one level directory
    """
    file_paths = glob.glob(dir_path + "/*") #create path to search for files.
    
    zf = zipfile.ZipFile(zip_path, 'w')
    dir_name = os.path.basename(dir_path)
    for p in file_paths:
        file_name = os.path.basename(p)
        zf.write(p, arcname=os.path.join(dir_name, file_name))
    zf.close()
    return zip_path




'''
BINNING
functions to help converte between floating point numbers and categories.
'''

def clamp(n, min, max):
    if n < min:
        return min
    if n > max:
        return max
    return n


'''
ANGLES
'''
def norm_deg(theta):
    while theta > 360:
        theta -= 360
    while theta < 0:
        theta += 360
    return theta

DEG_TO_RAD = math.pi / 180.0

def deg2rad(theta):
    return theta * DEG_TO_RAD

'''
VECTORS
'''
def dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))


def gather_tub_paths(cfg, tub_names=None):
    '''
    takes as input the configuration, and the comma seperated list of tub paths
    returns a list of Tub paths
    '''
    if tub_names:
        if type(tub_names) == list:
            tub_paths = [os.path.expanduser(n) for n in tub_names]
        else:
            tub_paths = [os.path.expanduser(n) for n in tub_names.split(',')]
        return expand_path_masks(tub_paths)
    else:
        paths = [os.path.join(cfg.DATA_PATH, n) for n in os.listdir(cfg.DATA_PATH)]
        dir_paths = []
        for p in paths:
            if os.path.isdir(p):
                dir_paths.append(p)
        return dir_paths


def gather_tubs(cfg, tub_names):    
    '''
    takes as input the configuration, and the comma seperated list of tub paths
    returns a list of Tub objects initialized to each path
    '''
    from .datastore import Tub
    
    tub_paths = gather_tub_paths(cfg, tub_names)
    tubs = [Tub(p) for p in tub_paths]

    return tubs

"""
Training helpers
"""

def get_image_index(fnm):
    sl = os.path.basename(fnm).split('_')
    return int(sl[0])


def get_record_index(fnm):
    sl = os.path.basename(fnm).split('_')
    return int(sl[1].split('.')[0])

def gather_records(cfg, tub_names, opts=None, verbose=False):

    tubs = gather_tubs(cfg, tub_names)

    records = []

    for tub in tubs:
        if verbose:
            print(tub.path)
        record_paths = tub.gather_records()
        records += record_paths

    return records

def get_model_by_type(model_type, cfg):
    '''
    given the string model_type and the configuration settings in cfg
    create a Keras model and return it.
    '''
    from .keras import  KerasLinear
 
    if model_type is None:
        model_type = cfg.DEFAULT_MODEL_TYPE
    print("\"get_model_by_type\" model Type is: {}".format(model_type))

    input_shape = (cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH)
    roi_crop = (cfg.ROI_CROP_TOP, cfg.ROI_CROP_BOTTOM)

    if model_type == "linear":
        kl = KerasLinear(input_shape=input_shape, roi_crop=roi_crop)
    else:
        raise Exception("unknown model type: %s" % model_type)

    return kl

def get_test_img(model):
    '''
    query the input to see what it likes
    make an image capable of using with that test model
    '''
    assert(len(model.inputs) > 0)
    try:
        count, h, w, ch = model.inputs[0].get_shape()
        seq_len = 0
    except Exception as e:
        count, seq_len, h, w, ch = model.inputs[0].get_shape()

    #generate random array in the right shape
    img = np.random.rand(int(h), int(w), int(ch))

    return img


def train_test_split(data_list, shuffle=True, test_size=0.2):
    '''
    take a list, split it into two sets while selecting a 
    random element in order to shuffle the results.
    use the test_size to choose the split percent.
    shuffle is always True, left there to be backwards compatible
    '''
    assert(shuffle==True)
    
    train_data = []

    target_train_size = len(data_list) * (1. - test_size)

    i_sample = 0

    while i_sample < target_train_size and len(data_list) > 1:
        i_choice = random.randint(0, len(data_list) - 1)
        train_data.append(data_list.pop(i_choice))
        i_sample += 1

    #remainder of the original list is the validation set
    val_data = data_list

    return train_data, val_data
    


"""
Timers
"""

class FPSTimer(object):
    def __init__(self):
        self.t = time.time()
        self.iter = 0

    def reset(self):
        self.t = time.time()
        self.iter = 0

    def on_frame(self):
        self.iter += 1
        if self.iter == 100:
            e = time.time()
            print('fps', 100.0 / (e - self.t))
            self.t = time.time()
            self.iter = 0
