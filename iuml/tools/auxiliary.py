'''
Auxilary functions for testing, logging, etc.
'''

from __future__ import print_function, division, absolute_import

import pandas as pd
import numpy as np

from iuml.tools.visual import *
from iuml.tools.train_utils import create_trainer
from iuml.training.segmentation import unet_proba_to_class_masks
from iuml.tools.clusters import *
from iuml.tools.image_utils import get_image_files

from datetime import datetime
from dateutil.parser import parse as parse_date_time

import os
import glob
import re

import requests
import cv2
import logging

from pymongo import MongoClient, ASCENDING, DESCENDING

def connect_prod_collection(env=None, **config):
    '''
    Connect prod collection (e.g.: sensordata)
    Parameters:
        env - name of the environment variable containing credentials in the form "username:password"
        **config - parameters:
            'mongo_connect' - connection format string
            'creds' - credentials
            'mongo_db' - db name
            'mongo_collection' - desired collection ('sensordata')
    Returns:
        pymongo client object pointing to the desired collection to be queried
    '''
    creds = os.environ[env] if env is not None else config['creds']

    mongo = MongoClient(config['mongo_connect'].format(creds))
    db = mongo[config['mongo_db']]
    sensordata = db[config['mongo_collection']]
    return sensordata

def create_logger(log_name = 'logger', log_file = None, console = True, suffix = None):
    '''
    Create a logger
    '''    

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s\t%(message)s')

    if log_file is not None:
        if not os.path.isdir(os.path.split(log_file)[0]):
            os.makedirs(os.path.split(log_file)[0])

        log_file, ext = os.path.splitext(log_file)
        if suffix is not None:
            log_file += datetime.now().strftime(suffix) + ext
        else:
            log_file += datetime.now().strftime("%y%m%d_%H%M%S") + ext


        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if console:
        cons_handler = logging.StreamHandler()
        cons_handler.setFormatter(formatter)
        logger.addHandler(cons_handler)

    return logger

def get_cv_data_for_date_range(title, facility, collection, date_range):
    '''
    Query a production collection for results within the specified date range
    Parameters:
        title - workflow title
        facility - facility
        collection - pymongo collection object
        daterange - (start, finish) datetime tuple
    '''
    start, now = date_range

    query = {'title': title, 
             'facility': facility, 
             'timestamp': {
                 '$gte': start,
                 '$lte': now
             }
            }

    df_res = pd.DataFrame(list(collection.find(query)))
    return df_res

def get_cv_datasets_for_date_range(title, facility, space, collection, date_range):
    '''
    Query a production collection for whole datasets within the specified date range
    Parameters:
        title - workflow title
        facility - facility
        space - space designation (e.g.: "house5")
        collection - pymongo collection object
        daterange - (start, finish) datetime tuple
    Returns:
        List of dataset names sorted by timestamp descending
    '''
    start, now = date_range

    query = {'title': title, 
             'facility': facility, 
             'space': space,
             'timestamp': {
                 '$gte': start,
                 '$lte': now
             }
            }
    
    all_datasets = list(collection.find(query))

    # timestamp pattern inside the dataset name
    pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z'
    timed_datasets = []
    for d in all_datasets:
        d['dataset_time'] = parse_date_time(re.findall(pattern, d['dataset_id'])[0])
        timed_datasets += [d]

    return pd.DataFrame(timed_datasets)

def get_image_urls_for_datasets(dataset_names, collection):
    '''
    Query a production collection for all images in given datasets
    Parameters:
        dataset_names - a list of datasets
        collection - pymongo collection object
    '''
    raise NotImplementedError("Not implemented")
    
def process_sensor_data(df_raw):

    '''
    Do some data manipulation
     
    1. Drop some columns
    2. Extract image urls from their list(dict) structure
    2. Map value to an actual numpy array

    Returns:
        zero_buds frame, non_zero_buds frame, customer name
    '''

    customer = df_raw.customer
    df = df_raw.drop(['__v', 'facility', 'sensorGroup', 'sensor_num', 'position', 'customer', 'role'], axis=1)
    df['images'] = df.loc[:, 'images'].apply(lambda im: im[0]['path'])
    df['value'] = df.loc[:, 'value'].apply(np.array)
    df['value_size'] = df.loc[:, 'value'].apply(get_num_buds_from_grid)
    df_zero_buds = df[df['value_size'] == 0]
    df_some_buds = df[df['value_size'] > 0]

    print("{} Entires with some buds present: {}".format(df_some_buds['space'].values[0], df_some_buds['_id'].count()))

    return df_zero_buds, df_some_buds, customer

def get_cv_records_by_file_names(title, facility, collection, file_list):
    '''
    Query a production collection for urls of given image file names
    Parameters:
        title - workflow title
        facility - facility
        collection - pymongo collection object
        file_list - list of image file names
    '''
    
    or_list = [{'name': {'$regex': f}} for f in file_list]
    query = {'title': title,
             'facility': facility,
             'images': {
                '$elemMatch': { '$or': or_list }
             }    
           }

    df_res = pd.DataFrame(list(collection.find(query)))
    return df_res

def get_recent_sample_recs(df, n=20):
    '''
    Retrieve n most recent records
    '''
    return df.reset_index(drop=True).sort_values('timestamp', ascending = False).loc[:n - 1, :]

def get_images(df, img_shape = None, out_dir = None):
    '''
    Loads resized images into a map of URL -> image
    '''
    ims = download_images(df['images'], img_shape, out_dir)
    url_img_map = {url: im for url, im in zip(df['images'], ims)}
    return url_img_map

def get_images_from_dir(title, facility, collection, dir, img_shape = None, out_dir = None):
    '''
    Download images stored in dir again (perhaps with a different size)
    '''

    file_list = [os.path.split(f)[1] for f in get_image_files(dir)]
    df_raw = get_cv_records_by_file_names(title, facility, collection, file_list)
    _, df, _ = process_sensor_data(df_raw)
    return get_images(df, img_shape, out_dir)

def get_image_masks(ims, model_file):
    tr = create_trainer('Unet', '')
    tr.model_file = model_file

    preds = tr.predict(data = ims)
    masks = unet_proba_to_class_masks(preds)
    return masks

def show_test_results(imgs, expected, actual, file_name = None):
    '''
    Show a grid of images with test results reflected
    Parameters:
        imgs - collection of images
        expected - collection of expected test results
        actual - collection of actual results
        file_name - (optional) file to save the visual to
    '''
    plt.rcParams['figure.figsize'] = 20, 20

    draw_grid_of_images(imgs, titles=lambda i: "{} expected {} actual {}".format(i, expected[i], actual[i]))
    if file_name is not None:
        plt.savefig(file_name)

def get_num_buds_from_grid(grid):
    '''
    Given a single grid retrieve number of buds
    '''
    if len(grid) == 0:
        return 0

    clusters = [v for v in np.array(grid).ravel() if len(v) > 0]
    if len(clusters) == 0:
        return 0
    return np.concatenate(clusters).size