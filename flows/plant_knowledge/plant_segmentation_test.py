from __future__ import print_function, division

import numpy as np
from iuml.tools.train_utils import create_trainer

import pandas as pd
import cv2
import os
import numpy as np
from  get_zones import *

from iuml.tools.image_utils import erode_with_mask, camera_from_file_name
from iuml.tools.visual import *
from iuml.tools.transforms import *
from iuml.tools.clusters import *
from iuml.training.segmentation import unet_proba_to_class_masks

import math

from get_zones import *
from plant_count import *
from sklearn.neighbors import NearestNeighbors

def refine_count(clusters, mask, zone_grid):
    '''
    Given cluster centroids with radii place them in the grid.
    '''

    zone_centroids = [get_center_of_zone(cell) for cell in zone_grid]
    knn = NearestNeighbors(n_neighbors=1).fit(zone_centroids)
    centers, radii = list(zip(*clusters))

    # find closest grid centroid
    distances, indicies = knn.kneighbors(centers)

    # build the map cell -> plant_center
    # eliminate false positives
    # based on pigeonhole principle: 
    # cannot have more plants than grid cells
    grid_code_book = np.zeros(len(zone_grid))
    for grid_idx in indicies.ravel():
        grid_code_book[grid_idx] = 1

    return grid_code_book, grid_code_book[grid_code_book > 0].size

def count_plants(df, df_camera, name, mask_orig_shape, expected):
    phase_1_counts = []
    final_counts = []

    mask, orig_shape = mask_orig_shape

    image_shape = mask.shape[::-1]

    print("Counts for {}".format(name))

    for name_case, df_case in df.groupby(level = ['File', 'Cam', 'Zone']):
        zone_array = df_camera.loc[name_case[1:]].values

        # extract the relevant mask and its roi
        roi_mask, roi = get_roi_mask(image_shape, orig_shape, mask, zone_array)
        
        # initial counting
        clusters, phase_1_num_plants, local_mask = get_plant_clusters(roi_mask, expected)

        phase_1_counts.append(phase_1_num_plants)

        # start refining:
        # get the new zone (relative to the lot we are interested in now)
        new_zone_array = transform_zone(zone_array, roi, orig_shape, image_shape)
        # get the grid for the zone - it's the trapezoid split into cells accordingly
        zone_grid = create_zone_grid(image_shape, (24, 5), new_zone_array)

        _, final_num_plants = refine_count(clusters, local_mask, zone_grid)
        final_counts.append(final_num_plants)
        
        print("\tCam: {}, Lot: {}, Phase 1: {}, Final: {}".format(name_case[1], name_case[2], phase_1_num_plants, final_num_plants))

    df = df.assign(Phase_1_Count = phase_1_counts, Final_Count = final_counts)    
    return df

def run_tests(zones_json, name_mask_map, test_setup_file, expected):
    '''
    Given test cases in the test_setup_file, run counting tests
    '''    
    # get camera setup
    image_descriptors = read_zones_from_json(zones_json)
    df_camera = image_descriptors_to_dataframe(image_descriptors)

    # get test setup
    df_test = pd.read_csv(test_setup_file, index_col=['File', 'Cam', 'Zone']).sort_index()
    df_counts = []
    total = len(name_mask_map.items())

    print("Starting counts...")
    
    for i, (name, df) in enumerate(df_test.groupby(level='File')):
        print("Counting file: {} of {}".format(i + 1, total))        
        mask_orig_shape = name_mask_map[name]

        actual_plant_counts = count_plants(df, df_camera, name, mask_orig_shape, expected)

        df_counts.append(actual_plant_counts)
        
    return pd.concat(df_counts)
    
def create_masks_for_dir(test_images_path, model_file, target_image_shape, n_classes = 2):
    '''
    Run Unet to get initial masks for all test images
    Returns:
        dict: file_name_without_extension -> (mask, WxH), where W x H are width and height of the original image
    '''
    trainer = create_counts_model(model_file, target_image_shape, n_classes)
    trainer.verbose = True

    unet_masks = trainer.predict(dir = test_images_path)
    masks = unet_proba_to_class_masks(unet_masks)
    files = glob.glob1(test_images_path, '*.jpg')
    paths = glob.glob(os.path.join(test_images_path, '*.jpg'))

    return {os.path.splitext(f)[0]: (mask, cv2.imread(p).shape[::-1][1:]) for f, mask, p in zip(files, masks, paths) }

if __name__ == "__main__":

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    target_image_shape = (816, 608)
    expected = 120

    zones_json = r'C:\Users\boris\Dropbox (Personal)\iunu\data\smith\thoth\cams\cam_positions.json'   
    test_setup_file = r'C:\Users\boris\Dropbox (Personal)\iunu\data\smith\thoth\cams\images\count_test\plant_counts.csv'
    test_images_path = r'C:\Users\boris\Dropbox (Personal)\iunu\data\smith\thoth\cams\images'
    model_file = r'C:\Users\boris\Dropbox (Personal)\iunu\data\smith\thoth\cnn\model_unet.hd5'
    
    test_results_file = os.path.join(os.path.split(test_setup_file)[0], 'segmentation_test_results.csv')
    name_mask_map = create_masks_for_dir(test_images_path, model_file, target_image_shape)

    df_phase_1 = run_tests(zones_json, name_mask_map, test_setup_file, expected)

    df_phase_1 = df_phase_1.reset_index()
    df_phase_1 = df_phase_1[['File', 'Cam', 'Zone', 'Plants', 'NIR', 'Expected', 'Phase_1_Count', 'Final_Count']]
    df_phase_1.to_csv(test_results_file, index=False)