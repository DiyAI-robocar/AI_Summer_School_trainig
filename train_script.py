#!/usr/bin/env python3

from PIL import Image
from tensorflow import ConfigProto, Session
from tensorflow.python import keras
from tensorflow.python.keras.layers import Convolution2D, Dropout, Flatten
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
import datetime
import glob
import json
import numpy as np
import os
import random
import time


def load_scaled_image_arr(filename, cfg):
    '''
    load an image from the filename, and use the cfg to resize if needed
    also apply cropping and normalize
    '''
    try:
        img = Image.open(filename)
        if img.height != cfg.IMAGE_H or img.width != cfg.IMAGE_W:
            img = img.resize((cfg.IMAGE_W, cfg.IMAGE_H))
        img_arr = np.array(img).astype(np.float32) / 255.0
    except Exception as e:
        print(e)
        print('failed to load image:', filename)
        img_arr = None
    return img_arr


def gather_tub_paths(cfg):
    '''
    takes as input the configuration, and the comma seperated list of tub paths
    returns a list of Tub paths
    '''
    paths = [os.path.join(cfg.DATA_PATH, n) for n in os.listdir(cfg.DATA_PATH)]
    dir_paths = []
    for p in paths:
        if os.path.isdir(p):
            dir_paths.append(p)
    return dir_paths


def gather_tubs(cfg):
    '''
    takes as input the configuration, and the comma seperated list of tub paths
    returns a list of Tub objects initialized to each path
    '''
    
    tub_paths = gather_tub_paths(cfg)
    tubs = [Tub(p) for p in tub_paths]

    return tubs


def get_image_index(fnm):
    sl = os.path.basename(fnm).split('_')
    return int(sl[0])


def get_record_index(fnm):
    sl = os.path.basename(fnm).split('_')
    return int(sl[1].split('.')[0])


def gather_records(cfg, verbose=False):
    tubs = gather_tubs(cfg)
    records = []

    for tub in tubs:
        if verbose:
            print(tub.path)
        record_paths = tub.gather_records()
        records += record_paths

    return records


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
    

class Tub(object):
    def __init__(self, path):

        self.path = os.path.expanduser(path)
        self.meta_path = os.path.join(self.path, 'meta.json')
        self.exclude_path = os.path.join(self.path, "exclude.json")
        self.df = None

        exists = os.path.exists(self.path)

        if exists:
            #load log and meta
            try:
                with open(self.meta_path, 'r') as f:
                    self.meta = json.load(f)
            except FileNotFoundError:
                self.meta = {'inputs': [], 'types': []}

            try:
                with open(self.exclude_path,'r') as f:
                    excl = json.load(f)
                    self.exclude = set(excl)
            except FileNotFoundError:
                self.exclude = set()

            try:
                self.current_ix = self.get_last_ix() + 1
            except ValueError:
                self.current_ix = 0

            if 'start' in self.meta:
                self.start_time = self.meta['start']
            else:
                self.start_time = time.time()
                self.meta['start'] = self.start_time

    def get_last_ix(self):
        index = self.get_index()           
        return max(index)

    def get_index(self, shuffled=True):
        files = next(os.walk(self.path))[2]
        record_files = [f for f in files if f[:6]=='record']
        
        def get_file_ix(file_name):
            try:
                name = file_name.split('.')[0]
                num = int(name.split('_')[1])
            except:
                num = 0
            return num

        nums = [get_file_ix(f) for f in record_files]
        
        if shuffled:
            random.shuffle(nums)
        else:
            nums = sorted(nums)
            
        return nums 

    def gather_records(self):
        ri = lambda fnm : int( os.path.basename(fnm).split('_')[1].split('.')[0] )

        record_paths = glob.glob(os.path.join(self.path, 'record_*.json'))
        if len(self.exclude) > 0:
            record_paths = [f for f in record_paths if ri(f) not in self.exclude]
        record_paths.sort(key=ri)
        return record_paths


# Override keras session to work around a bug in TF 1.13.1
# Remove after we upgrade to TF 1.14 / TF 2.x.
config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)
keras.backend.set_session(session)


class KerasPilot(object):
    '''
    Base class for Keras models that will provide steering and throttle to guide a car.
    '''
    def __init__(self, input_shape):
        self.optimizer = "adam"
        self.model = default_n_linear(2, input_shape)
        self.model.optimizer = keras.optimizers.Adam(lr=0.001, decay=0.0)
        self.compile()

    def load(self, model_path):
        self.model = keras.models.load_model(model_path)

    def load_weights(self, model_path, by_name=True):
        self.model.load_weights(model_path, by_name=by_name)
    
    def train(self, train_gen, val_gen, 
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):

        #checkpoint to save model after each epoch
        save_best = keras.callbacks.ModelCheckpoint(saved_model_path, 
                                                    monitor='val_loss', 
                                                    verbose=verbose, 
                                                    save_best_only=True, 
                                                    mode='min')
        
        #stop training if the validation error stops improving.
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                   min_delta=min_delta, 
                                                   patience=patience, 
                                                   verbose=verbose, 
                                                   mode='auto')
        
        callbacks_list = [save_best]

        if use_early_stop:
            callbacks_list.append(early_stop)
        
        hist = self.model.fit_generator(
                        train_gen, 
                        steps_per_epoch=steps, 
                        epochs=epochs, 
                        verbose=1, 
                        validation_data=val_gen,
                        callbacks=callbacks_list, 
                        validation_steps=steps*(1.0 - train_split))
        return hist

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss='mse')

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]


def adjust_input_shape(input_shape, roi_crop):
    height = input_shape[0]
    new_height = height - roi_crop[0] - roi_crop[1]
    return (new_height, input_shape[1], input_shape[2])


def default_n_linear(num_outputs, input_shape=(120, 160, 3), roi_crop=(0, 0)):

    drop = 0.1

    #we now expect that cropping done elsewhere. we will adjust our expeected image size here:
    input_shape = adjust_input_shape(input_shape, roi_crop)
    
    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu', name="conv2d_1")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu', name="conv2d_2")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu', name="conv2d_3")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu', name="conv2d_4")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu', name="conv2d_5")(x)
    x = Dropout(drop)(x)
    
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(drop)(x)

    outputs = []
    
    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear', name='n_outputs' + str(i))(x))
        
    return Model(inputs=[img_in], outputs=outputs)


def make_key(sample):
    tub_path = sample['tub_path']
    index = sample['index']
    return tub_path + str(index)


def collate_records(records, gen_records, opts):
    '''
    open all the .json records from records list passed in,
    read their contents,
    add them to a list of gen_records, passed in.
    use the opts dict to specify config choices
    '''

    new_records = {}
    
    for record_path in records:

        basepath = os.path.dirname(record_path)        
        index = get_record_index(record_path)
        sample = { 'tub_path' : basepath, "index" : index }
             
        key = make_key(sample)

        if key in gen_records:
            continue

        try:
            with open(record_path, 'r') as fp:
                json_data = json.load(fp)
        except:
            continue

        image_filename = json_data["cam/image_array"]
        image_path = os.path.join(basepath, image_filename)

        sample['record_path'] = record_path
        sample["image_path"] = image_path
        sample["json_data"] = json_data        

        angle = float(json_data['user/angle'])
        throttle = float(json_data["user/throttle"])

        sample['angle'] = angle
        sample['throttle'] = throttle

        sample['img_data'] = None

        # Initialise 'train' to False
        sample['train'] = False
        
        # We need to maintain the correct train - validate ratio across the dataset, even if continous training
        # so don't add this sample to the main records list (gen_records) yet.
        new_records[key] = sample
        
    # new_records now contains all our NEW samples
    # - set a random selection to be the training samples based on the ratio in CFG file
    shufKeys = list(new_records.keys())
    random.shuffle(shufKeys)
    trainCount = 0
    #  Ratio of samples to use as training data, the remaining are used for evaluation
    targetTrainCount = int(opts['cfg'].TRAIN_TEST_SPLIT * len(shufKeys))
    for key in shufKeys:
        new_records[key]['train'] = True
        trainCount += 1
        if trainCount >= targetTrainCount:
            break
    # Finally add all the new records to the existing list
    gen_records.update(new_records)


class MyCPCallback(keras.callbacks.ModelCheckpoint):
    '''
    custom callback to interact with best val loss during continuous training
    '''

    def __init__(self, send_model_cb=None, cfg=None, *args, **kwargs):
        super(MyCPCallback, self).__init__(*args, **kwargs)
        self.reset_best_end_of_epoch = False
        self.send_model_cb = send_model_cb
        self.last_modified_time = None
        self.cfg = cfg

    def reset_best(self):
        self.reset_best_end_of_epoch = True

    def on_epoch_end(self, epoch, logs=None):
        super(MyCPCallback, self).on_epoch_end(epoch, logs)
        '''
        when reset best is set, we want to make sure to run an entire epoch
        before setting our new best on the new total records
        '''        
        if self.reset_best_end_of_epoch:
            self.reset_best_end_of_epoch = False
            self.best = np.Inf
        

def on_best_model(model, model_filename):
    model.save(model_filename, include_optimizer=False)


def generator(opts, data, batch_size, isTrainSet=True):

    while True:
        batch_data = []
        keys = list(data.keys())
        random.shuffle(keys)
        kl = opts['keras_pilot']

        if type(kl.model.output) is list:
            model_out_shape = (2, 1)
        else:
            model_out_shape = kl.model.output.shape

        for key in keys:
            if not key in data:
                continue

            _record = data[key]

            if _record['train'] != isTrainSet:
                continue

            batch_data.append(_record)

            if len(batch_data) == batch_size:
                inputs_img = []
                angles = []
                throttles = []
                out = []

                for record in batch_data:
                    #get image data if we don't already have it
                    if record['img_data'] is None:
                        filename = record['image_path']

                        img_arr = load_scaled_image_arr(filename, cfg)

                        if img_arr is None:
                            break

                        if cfg.CACHE_IMAGES:
                            record['img_data'] = img_arr
                    else:
                        img_arr = record['img_data']
                    inputs_img.append(img_arr)
                    angles.append(record['angle'])
                    throttles.append(record['throttle'])
                    out.append([record['angle'], record['throttle']])

                if img_arr is None:
                    continue

                img_arr = np.array(inputs_img).reshape(batch_size, cfg.TARGET_H, cfg.TARGET_W, cfg.TARGET_D)
                X = [img_arr]

                if model_out_shape[1] == 2:
                    y = [np.array([out]).reshape(batch_size, 2) ]
                else:
                    y = [np.array(angles), np.array(throttles)]

                yield X, y

                batch_data = []


def train(cfg, model_name):
    '''
    use the specified data in tub_names to train an artifical neural network
    saves the output trained model as model_name
    '''
    verbose = cfg.VEBOSE_TRAIN

    model_type = 'linear'

    if model_name and not '.h5' == model_name[-3:]:
        raise Exception("Model filename should end with .h5")

    input_shape = (cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH)
    kl = KerasPilot(input_shape=input_shape)

    opts = { 'cfg' : cfg}
    opts['keras_pilot'] = kl

    print('training with model type', type(kl))

    kl.compile()

    if cfg.PRINT_MODEL_SUMMARY:
        print(kl.model.summary())

    records = gather_records(cfg, verbose=True)
    print('collating %d records ...' % (len(records)))

    gen_records = {}
    collate_records(records, gen_records, opts)

    train_gen = generator(opts, gen_records, cfg.BATCH_SIZE, True)
    val_gen = generator(opts, gen_records, cfg.BATCH_SIZE, False)

    total_records = len(gen_records)

    num_train = 0
    num_val = 0

    for key, _record in gen_records.items():
        if _record['train'] == True:
            num_train += 1
        else:
            num_val += 1

    print("train: %d, val: %d" % (num_train, num_val))
    print('total records: %d' %(total_records))

    steps_per_epoch = 100

    val_steps = num_val // cfg.BATCH_SIZE
    print('steps_per_epoch', steps_per_epoch)

    cfg.model_type = model_type

    start = time.time()

    model_path = os.path.expanduser(model_name)

    #checkpoint to save model after each epoch and send best to the pi.
    save_best = MyCPCallback(send_model_cb=on_best_model,
                             filepath=model_path,
                             monitor='val_loss',
                             verbose=verbose,
                             save_best_only=True,
                             mode='min',
                             cfg=cfg)

    #stop training if the validation error stops improving.
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=cfg.MIN_DELTA,
                                               patience=cfg.EARLY_STOP_PATIENCE,
                                               verbose=verbose,
                                               mode='auto')

    if steps_per_epoch < 2:
        raise Exception("Too little data to train. Please record more records.")

    epochs = cfg.MAX_EPOCHS

    workers_count = 1
    use_multiprocessing = False

    callbacks_list = [save_best]

    callbacks_list.append(early_stop)

    kl.model.fit_generator(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=cfg.VEBOSE_TRAIN,
        validation_data=val_gen,
        callbacks=callbacks_list,
        validation_steps=val_steps,
        workers=workers_count,
        use_multiprocessing=use_multiprocessing)

    duration_train = time.time() - start
    print("Training completed in %s." % str(datetime.timedelta(seconds=round(duration_train))) )

    print("\n\n----------- Best Eval Loss :%f ---------" % save_best.best)


class Config:
    def __init__(self):
        self.CAR_PATH = os.path.dirname(os.path.realpath(__file__))
        self.DATA_PATH = os.path.join(self.CAR_PATH, 'data')
        self.MODELS_PATH = os.path.join(self.CAR_PATH, 'models')

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
        self.CACHE_IMAGES = True             #keep images in memory. will speed succesive epochs, but crater if not enough mem.

        self.TARGET_H = self.IMAGE_H
        self.TARGET_W = self.IMAGE_W
        self.TARGET_D = self.IMAGE_DEPTH


if __name__ == '__main__':
    cfg = Config()
    model = "./models/mymodel.h5"

    train(cfg, model)
