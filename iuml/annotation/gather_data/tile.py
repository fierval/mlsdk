# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 23:42:37 2017

@author: iunu
"""

from __future__ import print_function, division

import os
from os import path
import cv2
import glob
import ntpath
import numpy as np
import time
from functools import partial
import multiprocessing as mp
import json

# Tile a single image - for multiprocessing
def tile_singe(img, file_name_out, length, cols, row_col):
    
    row, col = row_col
    
    im_cropped = img[row : row + length, col : col + length]
                
    r, c, _ = im_cropped.shape

    # resize to match the desired size                
    if r < length or c < length:
        fx = float(length) / im_cropped.shape[1]
        fy = float(length) / im_cropped.shape[0]
        im_cropped = cv2.resize(im_cropped, None, fx = fx, fy = fy)
    
    return im_cropped

def tile_and_resize_images(images_path, tiled_files_path, length=100, sample = 1., ext = '.jpg', prefix = ''):
    '''
    Creates tiles from image files, each tile is a square length x length
    Parameters:
        images_path - original images to be cropped
        tiled_files_path - path to tiles created
        length (default: 100) - length of a side of a tile
    Returns:
        None
    Remarks:
        Cropped images with dimensions less than "length"
        will be resized to length x length
    '''
    t1 = time.clock()

    orig_image_filter = os.path.join(images_path, '*' + ext)
    orig_image_files = glob.glob(orig_image_filter)
    
    if len(orig_image_files) == 0:
        raise ValueError("Source images don't exist")

    if not path.exists(tiled_files_path): 
        os.makedirs(tiled_files_path)
        
    for i, im_file in enumerate(orig_image_files):
        img = cv2.imread(im_file)
        if img is None:
            raise ValueError("Image not found")

        # the output file mask
        file_name_out = os.path.join(tiled_files_path, 
                                    os.path.splitext(os.path.split(im_file)[1])[0]) + prefix + '-{:d}.jpg'
        
        rows, cols, _ = img.shape
        cropped = [(row, col) for row in range(0, rows, length) for col in range(0, cols, length)]

        print("Tiling image {:d} of {:d}".format(i + 1, len(orig_image_files)))
        pool = mp.Pool()
        func = partial(tile_singe, img, file_name_out, length, cols)
        cropped = pool.map(func, cropped)
        pool.close()
        pool.join()

        # if we need to sample - sample!
        if sample < 1.0:
            sample_size = int(len(cropped) * sample)
            print("sampling {} images".format(sample_size))
            
            idxs = np.random.choice(len(cropped), sample_size, replace=False)
            cropped = [cropped[idx] for idx in idxs]
        
        print('Writing tiles for image {:d}'.format(i + 1))
        for k, im in enumerate(cropped):
            cv2.imwrite(file_name_out.format(k + 1), im)

    t2 = time.clock()
    print(t2 - t1)

def scale_marks(json_file, scale, out_file):
    '''
    Scales masks stored in the annotation file up or down
    '''
    if scale < 0 or scale == 1:
        raise ValueError("Scale should be positive, != 1")

    if not os.path.exists(json_file):
        raise ValueError('JSON file: {} does not exist'.format(json_file))

    with open(json_file) as in_file:
        image_descriptors = json.loads(in_file.read())

    for key, val in image_descriptors.items():
        for mark in val['marks']:
            mark['x'] *= scale
            mark['y'] *= scale

    with open(out_file, 'w') as outf:
        outf.write(json.dumps(image_descriptors, indent=4, separators=(',', ': ')))