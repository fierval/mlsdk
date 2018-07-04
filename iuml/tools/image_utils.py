from __future__ import absolute_import, division, print_function

import cv2
import glob
import os
from . import validate
import numpy as np
import re

def image_exts():
    return ['*.jpg', '*.png']

def camera_from_file_name(fn):
    pattern = r'cam0[0-9]{3}'
    return int(re.findall(pattern, fn, re.IGNORECASE)[0][3:])

def get_image_files(path):
    files = []
    for ext in image_exts():
        files += glob.glob(os.path.join(path, ext))

    return files

def copy_with_resize(src, dest, size = None, ratio = None):
    '''
    Copy and image while resizing it if necessary

    Parameters
        src - source path
        dest - destination path
        size = (width, height) target size
        ratio = (width_ratio, height_ratio)
    '''

    validate.raise_if_not_exists(src)
    validate.create_if_not_exists(dest)

    if size is None and ratio is None:
        raise ValueError('either size or ratio must be specified')

    files = get_image_files(src)
    total = len(files)

    i = 0
    for f in files:
        img = cv2.imread(f)

        fx, fy = ratio
        img = cv2.resize(img, size, fx = fx, fy = fy)

        out_file = os.path.join(dest, os.path.split(f)[1])
        cv2.imwrite(out_file, img)
        i += 1
        if i % 10 == 0:
            print("copied {} of {} files".format(i, total))

    if i % 10 > 0:
        print("Complete.")

def convert_mask_fiji_to_class(src, dest):
    '''
    Converts a directory of masks created by Fiji to
    a format expected by the SDK  (i.e.: each entry is a class)
    handling n-classes == 2 where green == 0 (background), red == 1
    '''
    validate.raise_if_not_exists(src)
    validate.create_if_not_exists(dest)

    files = get_image_files(src)
    total = len(files)

    i = 0
    for f in files:
        img = cv2.imread(f)

        # remove the blue channel
        img2 = np.delete(img, 0, 2)

        # classes are assumed to be: green - 0, red - 1
        img_gray = np.argmax(img2, axis = 2)

        out_file = os.path.join(dest, os.path.split(f)[1])
        cv2.imwrite(out_file, img_gray)

        i += 1
        if i % 10 == 0:
            print("copied {} of {} files".format(i, total))

    if i % 10 > 0:
        print("Complete.")

from operator import itemgetter
from sklearn.cluster import k_means

def convert_mask_fiji_to_classes_codebook(gray, classes):
    '''
    Given a gray-scaled mask, convert it to our classes representation.
    Parameters:
        gray - the mask in grayscale
        classes - order of classes on the color scale
    '''
    # histogram of colors. See above: peaks are at dry, background, moist
    # in this order
    hist, bins = np.histogram(gray.ravel(), 256, [0, 256])
    n_classes = len(classes)

    # cluster and compute codebook
    colors = np.nonzero(hist)[0]
    _, cls, _ = k_means(np.reshape(colors, (-1, 1)), n_classes)
    starts = [np.argwhere(cls == i).reshape(-1) for i in range(n_classes)]
    starts = sorted(starts, key=itemgetter(0))

    # map codebook to actual classes
    for i, c in enumerate(classes):
        cls[starts[i]] = c

    mask = gray.copy()
    for c, classs in zip(colors, cls):
        mask[mask == c] = classs
    return mask

def erode_with_mask(gray, mask, kernel=(3,3), iterations = 1):
    '''
    Erode a gray-scale image given a mask
    Parameters:
        gray - original image
        mask - mask where to apply erosion. May be of any type, not just uint8
        kernel - erosion kernel
        iterations - number of iterations

    Returns:
        Copy of the original image with the masked region eroded
    '''

    im = np.where(mask > 0, np.ones_like(mask), np.zeros_like(mask)).astype(np.uint8)

    im = cv2.erode(im, np.ones(kernel), iterations = iterations)

    res = gray.copy()
    res[mask > 0] = im[mask > 0]
    return res
