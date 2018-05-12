from __future__ import print_function, division, absolute_import

import numpy as np

from iuml.tools.visual import *
from iuml.tools.clusters import *
from iuml.tools.auxiliary import *

from datetime import datetime, timedelta
import os
import time

import cv2

def run_test(config, common, logger):

    model_file = config['model_file']
    sensordata = connect_prod_collection(**common)

    date_time_now = datetime.now()
    dtnowstr = date_time_now.strftime(common['current_time_suffix'])

    # query parameters
    title = config['title']
    facility = config['facility']

    days = config['days_delta']
    sample_size = config['sample_size']
    if sample_size <= 30:
        splt = os.path.splitext(config['out_file'])
        name, ext = splt
        out_file = (name + '{}{}').format(dtnowstr, ext)
        print(out_file)
    else:
        out_file = None

    imgs_dir = None
    save_images = False

    if "images_dir" in config.keys():
        imgs_dir = config["images_dir"] + dtnowstr
        save_images = True

        if not os.path.exists(imgs_dir):
            os.makedirs(imgs_dir)

    img_shape = None
    if "img_shape" in config.keys():
        img_shape = eval(config['img_shape'])

    start_test = time.time()
    starting_date_time_str = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    logger.info("Starting: {}".format(starting_date_time_str))

    now = datetime.utcnow()
    startDate = now - timedelta(days=days)
    nowStr = now.strftime('%m/%d/%Y')
    startStr = startDate.strftime('%m/%d/%Y')

    logger.info("Connecting and getting cv data {} - {}".format(startStr, nowStr))

    start = time.time()
    df_raw = get_cv_data_for_date_range(title, facility, sensordata, (startDate, now))
    logger.info("Retrieved cv data: {:.2f}s".format(time.time() - start))

    logger.info("Retrieving most recent {} samples".format(sample_size))
    df_zero, df_nonzero, customer = process_sensor_data(df_raw)
    if not 'activation' in config.keys() or config['activation'] == 'nonzero':
        df_samples = get_recent_sample_recs(df_nonzero, sample_size)
    else:
        df_samples = get_recent_sample_recs(df_zero, sample_size)


    # retrieve images
    start = time.time()

    if not save_images:    
        imgs_dir = None

    url_img_map = get_images(df_samples, img_shape, out_dir = imgs_dir)
    ims = list(url_img_map.values())
    logger.info("Downoaded {} images in {:.2f}s".format(sample_size, time.time() - start))

    # run through the model and get expected results
    if config['download_only']:
        logger.info("Download only. Finishing up.")
        return

    start = time.time()
    masks = get_image_masks(ims, model_file)
    logger.info("Created inferences for downloaded images in {:.2f}s".format(time.time() - start))

    clusters = [get_clusters_from_mask(m, eps=5, min_samples=5) for m in masks]
    threshold = config['threshold']
    cluster_sizes_expected = [0 if c is None or len(c)  <= threshold else len(c) for c, _ in clusters]
    cluster_sizes_actual = get_cluster_sizes_actual(df_samples, url_img_map)
    logger.info("Computed clusters")    

    # get avg cumulative error
    mean_error_vals = [abs(exp - act) for exp, act in zip (cluster_sizes_expected, cluster_sizes_actual)]
    show_test_results(ims, cluster_sizes_expected, cluster_sizes_actual, file_name=out_file)

    if np.mean(mean_error_vals) <= 1.0 or np.max(mean_error_vals) <= 3:
        logger.info("PASSED")
    else:
        logger.error("FAILED")
        for exp, act in zip (cluster_sizes_expected, cluster_sizes_actual):
            logger.info("Expectd: {}, Actual: {}".format(exp, act))

    logger.info("Finished in {:.2f}s".format(time.time() - start_test))

def view_sample_image(url_img_map, masks, idx):
    '''
    Display an image and its mask by index
    '''
    plt.figure(figsize= (20, 20))
    show_image_and_mask(url_img_map[list(url_img_map.keys())[idx]], masks[idx])

def get_cluster_sizes_actual(df, url_img_map):

    sizes = []
    for url in url_img_map.keys():
        val = df[df['images'] == url].value.values[0]
        clusters = [v for v in val.ravel() if len(v) > 0]
        if(len(clusters) > 0):
            clusters = np.concatenate(np.array(clusters))
        sizes.append(len(clusters))
    return sizes

def show_clusters(img, centers, radii, img_shape = (816, 608)):
    im = cv2.resize(ims[idx].copy(), img_shape)
    centers = list(map(tuple, centers))

    for c, r in zip(centers, radii):
        _ = cv2.circle(im, c, r, (255, 255, 0), 3)
    show_image(im)    
