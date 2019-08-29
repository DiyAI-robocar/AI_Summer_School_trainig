#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 12:32:53 2017

@author: wroscoe
"""
import os
import time
import json
import random
import glob


class Tub(object):
    """
    A datastore to store sensor data in a key, value format.

    Accepts str, int, float, image_array, image, and array data types.

    For example:

    #Create a tub to store speed values.
    >>> path = '~/mydonkey/test_tub'
    >>> inputs = ['user/speed', 'cam/image']
    >>> types = ['float', 'image']
    >>> t=Tub(path=path, inputs=inputs, types=types)

    """

    def __init__(self, path, inputs=None, types=None, user_meta=[]):

        self.path = os.path.expanduser(path)
        #print('path_in_tub:', self.path)
        self.meta_path = os.path.join(self.path, 'meta.json')
        self.exclude_path = os.path.join(self.path, "exclude.json")
        self.df = None

        exists = os.path.exists(self.path)

        if exists:
            #load log and meta
            #print("Tub exists: {}".format(self.path))
            try:
                with open(self.meta_path, 'r') as f:
                    self.meta = json.load(f)
            except FileNotFoundError:
                self.meta = {'inputs': [], 'types': []}

            try:
                with open(self.exclude_path,'r') as f:
                    excl = json.load(f) # stored as a list
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

        elif not exists and inputs:
            print('Tub does NOT exist. Creating new tub...')
            self.start_time = time.time()
            #create log and save meta
            os.makedirs(self.path)
            self.meta = {'inputs': inputs, 'types': types, 'start': self.start_time}
            for kv in user_meta:
                kvs = kv.split(":")
                if len(kvs) == 2:
                    self.meta[kvs[0]] = kvs[1]
                # else exception? print message?
            with open(self.meta_path, 'w') as f:
                json.dump(self.meta, f)
            self.current_ix = 0
            self.exclude = set()
            print('New tub created at: {}'.format(self.path))
        else:
            msg = "The tub path you provided doesn't exist and you didnt pass any meta info (inputs & types)" + \
                  "to create a new tub. Please check your tub path or provide meta info to create a new tub."

            raise AttributeError(msg)


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

