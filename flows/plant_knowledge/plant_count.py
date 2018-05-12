from __future__ import print_function, division

import numpy as np
from iuml.tools.train_utils import create_trainer

import cv2
import os
import numpy as np
from  get_zones import *

from iuml.tools.image_utils import erode_with_mask, camera_from_file_name
from iuml.tools.transforms import *
from iuml.tools.clusters import *
from iuml.training.segmentation import unet_proba_to_class_masks

import math

def create_counts_model(model_file, image_shape, n_classes = 2):

    params = dict(weights_file = model_file, n_classes=n_classes, img_shape = image_shape)

    # create the trainer, since we aren't using it
    # for training pass "" for path.
    # Can't pass None because of some path.join operations
    return create_trainer('Unet', "", **params)

def count_fixed_plants(img, model_file, image_shape, zone_array, num_expected_plants, trainer = None):
    '''
    Count plants in a fixed zone
    img -- numpy array representing the image (BGR)
    model_file - model weights for Unet
    zone_array - polygon designating the zone of interest
    num_expected_plants - how many plants are we expecting
    '''

    if not os.path.exists(model_file):
        raise FileExistsError("File does not exist: {}".format(model_file))

    # image_shape: tuple of W x H (col x rows) for an image
    
    mask = get_predictions(img, model_file, image_shape, trainer)

    # we are going to be reshaping the mas
    final_mask, _ = get_roi_mask(image_shape, img.shape[:2][::-1], mask, zone_array)

    clusters, num_plants, mask = get_plant_clusters(final_mask, num_expected_plants)
    return clusters, num_plants, mask

def get_predictions(img, model_file, image_shape, trainer):

    if trainer is None:
        trainer = create_counts_model(model_file, image_shape)

    preds = trainer.predict(data=[img])
    masks = unet_proba_to_class_masks(preds)

    # since there is only one image - we just unpack the list
    return masks[0]


def get_plant_clusters(mask, num_expected):
    '''
    Parameters:
        mask - masked image from which clusters are extracted
        num_expected - number of expected plants
    Returns:
        array of tuples (center, radius) designating segmented plants
        num_plants - number of plants we are counting
    '''

    # initial clustering
    centers, radii, coords, clusters, n_clusters, avg_radius = get_clusters_from_mask(mask)

    if (n_clusters == num_expected):
        return np.array(list(zip(centers, radii))), num_expected, mask
    
    # REVIEW: if we can barely see things...
    if (avg_radius <= 3):
        return np.array(list(zip(centers, radii))), num_expected, mask

    kernel = (3, 3)

    # fused plant circles are very likely.
    # REVIEW: heuristic for the anomalous "fused" clusters
    fused_clusters = np.where(radii >= 2 * avg_radius)[0]
    
    n_loops = 15
    while fused_clusters.size > 0 and n_loops > 0:
        n_loops -= 1
        # this is a mask where each plant (or "fused" plants) has its own
        # unique cluster assignment
        clusters_mask = np.zeros(mask.shape)
        # so that we can distinguish clusters from background
        # we shift their indicies by 1
        clusters_mask[tuple(zip(*coords))] = clusters + 1

        # REVIEW: this loop is easily parallelized
        # if we return a different mask from erolde_with_mask
        # and then "&" them all with the original
        # We loop over "fused clusters" and try to unfuse them
        # with a (3, 3) erosion kernel
        for fused in fused_clusters:
            cur_mask = np.where(clusters_mask == fused + 1,
                            np.ones_like(mask),
                            np.zeros_like(mask)).astype(np.uint8)

            mask = erode_with_mask(mask, cur_mask, kernel)

        centers, radii, coords, clusters, n_clusters, avg_radius = get_clusters_from_mask(mask)
        fused_clusters = np.where(radii >= 2 * avg_radius)[0]

    return np.array(list(zip(centers, radii))), n_clusters, mask

def get_zoi(img_file, cams, zone=0):
    image_file_name = os.path.split(img_file)[1]
    camera = camera_from_file_name(image_file_name)

    # see if we got a non-existent camera
    try:
        zone_array = cams.loc[camera].values
    except :
        raise ValueError("No zone mapping exists for camera: {}".format(camera))

    return zone_array

if __name__ == '__main__':
    import glob
    
    # older Titan quicker to initialize
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    image_shape = (816, 608)

    test_images_path = r'C:\Users\boris\Dropbox (Personal)\iunu\data\smith\thoth\cams\images'
    zones_json = r'C:\Users\boris\Dropbox (Personal)\iunu\data\smith\thoth\cams\cam_positions.json'   
    model_file = r'C:\Users\boris\Dropbox (Personal)\iunu\data\smith\thoth\cnn\model_unet.hd5'

    test_images_files = glob.glob(os.path.join(test_images_path, "*.jpg"))

    cam = 'cam0720'
    image_file = [f for f in test_images_files if cam in f][0]
    img = cv2.imread(image_file)

    expected = 120
    image_descriptors = read_zones_from_json(zones_json)
    camera_zones_frame = image_descriptors_to_dataframe(image_descriptors)

    zone_array = get_zoi(image_file, camera_zones_frame, zone=2)

    clusters, num_plants, mask = count_fixed_plants(img, model_file, image_shape, zone_array, expected)

    print("Counted {} plants".format(num_plants))
