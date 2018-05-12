
from __future__ import print_function, division
import cv2
import numpy as np
from numba import jit

@jit
def copy_with_resize(src, dest, size):
    '''
    Copy an image with resizing
    Parameters:
        src - source
        dest - destination
        size - (width, height) tuple
    '''
    
    img = cv2.imread(src)
    if img.shape[:2][::-1] != size:
        img = cv2.resize(img, size)
    cv2.imwrite(dest, img)

@jit    
def mask_from_image_to_unet_label_fast(mask, n_classes):

    labels = np.zeros(mask.shape + (n_classes,), dtype=np.uint8)

    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            plane = mask[r,c]

            # REVIEW: masks that come out of Weka may have junk in them!!
            if plane >= n_classes:
                if n_classes == 2:
                    labels[r, c, 1] = 1
                else:
                    labels[r, c, 0] = 1
            else:                        
                labels[r, c, plane] = 1
    return labels

def mask_from_image_to_unet_label(mask_src, mask_dest, size, n_classes):
    '''
    Convert mask from image to a numpy array usable for Unet
    Parameters:
        mask_src - source mask file (image)
        mask_dest - destination file - saved numpy array
        size - (width, height) tuple
        n_classes - number of classes
    '''
    
    mask = cv2.imread(mask_src, cv2.IMREAD_GRAYSCALE)
    if mask.shape[:2][::-1] != size:
        mask = cv2.resize(mask, size)

    labels = mask_from_image_to_unet_label_fast(mask, n_classes)

    np.savez_compressed(mask_dest, labels=labels)

def unet_proba_to_class_masks(labels):
    '''
    Convert each entry in the label array of predictions from probability
    to actual class-number mask
    Parameters:
        labels -- N x H x W x C (C - number of classes) numpy array
    Returns:
        N x H x W numpy array of masks
    '''

    if len(labels.shape) < 4:
        return labels

    return np.argmax(labels, axis=3)
