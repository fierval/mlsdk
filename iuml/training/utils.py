# -*- coding: utf-8 -*-

from __future__ import print_function, division
from tensorflow.python.client import device_lib
import re
import os
import glob
from sklearn.model_selection import ShuffleSplit
import shutil
import keras
import json
import cv2
from iuml.tools.image_utils import get_image_files
import numpy as np

def get_num_of_gpus():
    '''
    Returns True if Tensorflow can detect a GPU device
    '''
    devices = [dev.name for dev in device_lib.list_local_devices()]
    
    total = 0
    for d in devices:
        matches = re.findall(r'gpu', d, re.IGNORECASE)
        if len(matches) > 0: total += 1
    
    print("Found total {} GPUs".format(total))
    return total

def has_gpu():
    return get_num_of_gpus() > 0

def clean_training_validation_trees(train_data_root, validation_data_root):
    if os.path.exists(validation_data_root):
        print('removing validation...')
        shutil.rmtree(validation_data_root)
    
    os.makedirs(validation_data_root)
    
    if os.path.exists(train_data_root):
        print('removing training...')
        shutil.rmtree(train_data_root)

    os.makedirs(train_data_root)

def split_training_validation_data(data_root, out_data_root, validate_fraction=0.1):
    '''
    Generates validation data from a given root folder, under which subfolders hold
    files of different classes

    Returns:
        (training_size, validation_size)

    '''

    train_data_root = os.path.join(out_data_root, 'training')
    validation_data_root = os.path.join(out_data_root, 'validation')
    
    if not (0 < validate_fraction < 1.0):
        raise ValueError("validate_to_train_ration should be between 0.0 and 1.0" )
        
    clean_training_validation_trees(train_data_root, validation_data_root)
        
    subfolder_candidates = glob.glob(os.path.join(data_root, "*"))
    subfolders = [d for d in subfolder_candidates if os.path.isdir(d)]
    class_names = [os.path.split(cn)[1] for cn in subfolders]
    
    if len(subfolders) == 0:
        raise ValueError('No subfolders found. Perhaps {} does not exist'.format(train_data_root))
        
    # This shuffles and then splits datasets
    # the output is indexes into the dataset being split
    rs = ShuffleSplit(n_splits=1, test_size=validate_fraction)
    
    print("Creating splits...")

    total_train = 0
    total_valid = 0

    for s, cn in zip(subfolders, class_names):
        files = glob.glob1(s, '*.*')
        in_files = glob.glob(os.path.join(s, '*.*'))
        
        splits = list(rs.split(files))[0]

        total_train += splits[0].shape[0]
        total_valid += splits[1].shape[0]

        for dataset_path, split, name in zip([train_data_root, validation_data_root], splits, ['training', 'validation']):        
            out_path = os.path.join(dataset_path, cn)
            
            if not os.path.exists(out_path):
                os.makedirs(out_path)
                
            out_files = [os.path.join(out_path, files[idx]) for idx in split]
            
            print('Creating {} dataset for class: {}'.format(name, cn))
            
            for inf, outf in zip(in_files, out_files):
                shutil.copyfile(inf, outf)

    return (total_train, total_valid)

def freeze_batch_normalization_layers(model):
    '''
    Freeze batch normalization layers for segmentation
    '''

    # freeze batch normalizatin layers
    for layer in model.layers:
        layer.trainable = False
        if isinstance(layer, keras.layers.normalization.BatchNormalization):
            layer._per_input_updates = {}

def create_preprocessing_config(config_file, images_dir):
    '''
    Compute per-layer averages and stds for a set of images.
    Store them in the json config file
    Parameters:
        config_file - json file to which means and stds will be written
        images_dir - directory of images to compute normalization parameters

    Returns:
        A dictionary: 'means': [], 'stds': [] of normalization parameters. One per BGR layer.
        The dictionary is serialized to JSON
    '''

    if os.path.exists(config_file):
        os.remove(config_file)

    if not os.path.exists(images_dir):
        raise FileNotFoundError('images directory does not exist: {}'.format(images_dir))

    if not os.path.isdir(images_dir):
        raise NotADirectoryError('specified path is not a directory: {}'.format(images_dir))

    # read them all and stack'em up for easy computation
    image_files = get_image_files(images_dir)
    ims = [np.expand_dims(cv2.imread(f), axis=0) for f in image_files]
    ims = np.vstack(ims)

    print("loaded {} images".format(len(ims)))
    means = np.mean(ims, axis=(0, 1, 2))
    stds = np.std(ims, axis=(0, 1, 2))
    
    print("computed means & stds")

    normalization_dictionary = {'means': list(np.squeeze(means)), 'stds': list(np.squeeze(stds))}

    # store them in the json file
    with open(config_file, 'w') as conf:
        json.dump(normalization_dictionary, conf)

    print("saved {}".format(config_file))

    return transform_normalization_dictionary(normalization_dictionary)


def load_normalization_dictionary(config_file):
    '''
    Load previously saved normalization dictionary
    '''

    dict = {}
    if not os.path.exists(config_file):
        raise FileNotFoundError("File not found: {}".format(config_file))

    with open(config_file, 'r') as conf:
        dict = json.load(conf)
    
    return transform_normalization_dictionary(dict)

def transform_normalization_dictionary(dict):
    '''
    Transform each of the means and stds array from shape (3,) to shape (3, 1, 1)
    Parameters:
        dict - dictionary: {'means': means_array, 'stds': stds_array}
    Returns:
        dictionary with arrays transformed
    '''

    dict['means'] = np.array(dict['means'], np.float32)
    dict['stds'] = np.array(dict['stds'], np.float32)
    return dict
