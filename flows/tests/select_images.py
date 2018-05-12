'''
# coding: utf-8
Based on https://github.com/iunullc/machine-learning-sdk/blob/master/docs/model_testing_methodology.md
'''

import os
import iuml.tools.auxiliary as iutools
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from multiprocessing import pool
import multiprocessing
from datetime import datetime

from iuml.tools.train_utils import create_trainer
from iuml.tools.visual import *
from iuml.tools.net import download_images
from iuml.tools.image_utils import *

def run_test(config, common, logger):
    root_path = config['root_path']

    # Need a `config` object which is a dictionary:
    # {
    #  'creds': 'username:password'
    #  'mongo_collection': 'sensordata',
    #  'mongo_connect': 'mongodb://{}@candidate.21.mongolayer.com:11857/app80974971',
    #  'mongo_db': 'app80974971'
    # }
    # Alternatively `creds` doesn't have to be present, then an environment variable with credentials should be passed:
    # iutools.connect_prod_collection(env='MONGO_CREDS', **config)

    collection = iutools.connect_prod_collection(**common)


    # ### Parameterize Query ###
    # 
    # title, facility, spaces - fixed  
    # threshold - what is the minimum number of datasets extracted over the given date range to be considered testable  
    # days - over how many days (starting @ midnight of the next day)

    # Relevant query fields
    title = 'CV_bud_grid'
    facility = 'sanysabel'
    spaces = config['spaces']

    # collect over...
    days = config['days']


    # ### Fix Date Range ###
    now = datetime.utcnow() + timedelta(days=1)
    # set to the midnight of the next day
    endtime = datetime(now.year, now.month, now.day)
    date_range = (now - timedelta(days=days), endtime)


    # ### Query for the desired date range ###
    def retrieve_dataset(space):
        df_raw = iutools.get_cv_datasets_for_date_range(title, facility, space, collection, date_range)
        if df_raw.size == 0:
            return None
        df_zero_buds, df_nonzero_buds, _ = iutools.process_sensor_data(df_raw)
        df = pd.concat([df_nonzero_buds, df_zero_buds], ignore_index=True)
        df['num_buds'] = df['value_size']
        df.drop(['_id', 'title', 'value_size'], inplace=True, axis=1)
        return df

    # Retrieve datasets in the desired range from the DB
    print("Running query on {} days of datasets".format(days))

    retriever = pool.ThreadPool(multiprocessing.cpu_count())
    datasets = retriever.map(retrieve_dataset, spaces)    
    datasets = [d for d in datasets if d is not None]
    if len(datasets) == 0:
        print("Nothing could be retrieved")
        return
    
    # filter out images    
    dfs = pd.concat(datasets)
    dfs['hour'] = dfs['timestamp'].dt.hour
    dfs[dfs['hour'] == 0] = 24
    dfs = dfs.set_index(['space', 'dataset_id'])
    start_hour, end_hour = eval(config['filter_utc_hours'])
    if start_hour >= end_hour:
        raise ValueError("start hour shoulr be greater than end hour")

    # Filter out by UTC time (e.g: 3, 23)
    dfs = dfs[dfs["hour"] >= start_hour]
    dfs = dfs[dfs["hour"] <= end_hour]

    # ### Aggregate by Space & Dataset ###
    group = dfs.groupby(level=[0, 1])
    d_agg_count = group['num_buds'].agg({'total': 'sum', 'count' : 'count'})
    metadata = group[['dataset_time']].first()
    merged = d_agg_count.merge(metadata, left_index=True, right_index=True)
    

    # ### Create trainer & instantiate model###
    # We need image shape
    net = 'Unet'
    batch_size = 1

    model_file_name = os.path.join(root_path, config['model_file'])

    params = dict(batch_size = batch_size, model_file=model_file_name)

    # not using the trainer for actual training, pass '' for root_train parameter
    trainer = create_trainer(net, '', **params)
    trainer.instantiate_model(predict=True)
    input_shape = trainer.model.input_shape[1:3][::-1]

    # ### Sample URLs from each dataset ###
    # 
    # A directory will be created: "sample_&lt;current date time &gt;" and everything will be saved there:
    # - images
    # - test configuratin data

    total_samples = config['total_samples']
    samples_per_dataset = config['samples_per_dataset']
    total_datasets = int(np.ceil(total_samples / samples_per_dataset))

    have_spaces = set(map(lambda v: v[0], merged.index.values))

    datasets_per_space = int(np.ceil(total_datasets / len(have_spaces)))
    print("total datasets: {}, datasets_per_space: {}".format(total_datasets, datasets_per_space))

    sampled = []
    for space in have_spaces:
        # get a list of datasets
        df_merged_space = merged.loc[space]
        dataset_ids =  df_merged_space.index
        if total_datasets - datasets_per_space >= 0:
            sample_size = datasets_per_space
            total_datasets -= datasets_per_space
        else:
            sample_size = total_datasets
    
        keep_sampling = True
    
        # sometimes, there are fewer datasets than sample size!
        while keep_sampling and sample_size > 0:
            try:
                dataset_ids = np.random.choice(dataset_ids, size=sample_size, replace=False)
                keep_sampling = False
            except:
                sample_size -= 1

        # from each datset, sample images
        if sample_size > 0:
            im_samples = []
            for d_idx in dataset_ids:
                try:
                    im_samples.append(dfs.loc[(space, d_idx)].sample(n=samples_per_dataset))
                except:
                    continue    
            # flatten
            if len(im_samples) > 0:
                sampled.append(pd.concat(im_samples))

    if len(sampled) == 0:
        print("nothing sampled")
        return

    df_sampled = pd.concat(sampled)

    # ### Output the Test Configuration File ###
    # #### Download Samples  ####
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    sampled_path = os.path.join(root_path, r'sampled_{}'.format(now))
    if not os.path.exists(sampled_path):
        os.makedirs(sampled_path)

    print("Starting async download of images... to {}".format(sampled_path))
    images = download_images(df_sampled.loc[:, 'images'], img_shape = input_shape,  out_dir=sampled_path)
    
    # #### Validate download and remove entries that did not download correctly ####
    bad_idxs = [i for i, im in enumerate(images) if len(np.nonzero(im)[0]) == 0]

    df_reset = df_sampled.reset_index()
    df_reset = df_reset.drop(bad_idxs)
    df_sampled = df_reset.set_index(['space', 'dataset_id'])    
    df_sampled.shape

    # #### Create Test Config and Serialize ####
    print("Creating test config in: {}".format(sampled_path))

    sampled_list = []
    for space in have_spaces:
        try:
            for ds_id in set(df_sampled.loc[space].index):

                total_buds_predicted = merged.loc[(space, ds_id)]['total']
                dataset_size = merged.loc[(space, ds_id)]['count']
                dataset_time = merged.loc[(space, ds_id)]['dataset_time'].strftime('%Y-%m-%d %H:%M:%S')
        
                # get images
                sampled = df_sampled.loc[(space, ds_id)]
                sampled['file'] = sampled['images'].map(lambda x: x[x.rfind("/") + 1: ])
        
                sampled['total_predicted'] = total_buds_predicted
                sampled['dataset_time'] = dataset_time
                sampled['dataset_size'] =dataset_size
                sampled = sampled.drop('value', axis=1)
                sampled_list.append(sampled)
        except:
            continue

    df_sampled = pd.concat(sampled_list)        
    test_config_file = os.path.join(sampled_path, 'test_config.csv')    
    df_sampled.to_csv(test_config_file)
