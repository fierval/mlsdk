from __future__ import division, print_function

import os
import cv2
import numpy as np
import copy
import math
import multiprocessing as mp
from iuml.tools.clusters import *
from iuml.tools.image_utils import get_image_files

import random
import time

import glob

class ClusterViewer(object):
    '''
    Compare actual images to images with cluster overlay
    '''

    def __init__(self, im_dir, mask_dir):
        if not os.path.exists(im_dir) or not os.path.exists(mask_dir):
            raise ValueError("images or masks dir does not exist")

        self.im_files = get_image_files(im_dir)
        self.im_window = 'Image'
        self.mask_window = 'Overlay'
        self.mask_dir = mask_dir
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.total = len(self.im_files)

        self.display_text = True

    def display_image_and_overlay(self, im_file, i):
        '''
        Return the image and its "overalyed" clusters
        '''
        img = cv2.imread(im_file)

        fn = os.path.split(im_file)[1]

        mask_file = os.path.join(self.mask_dir, fn)
        
        # some masks may have already been munged
        green = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
              
        green = cv2.GaussianBlur(green, (5, 5), 0)

        blue = np.zeros_like(green)
        red = np.ones_like(green)
        red [green > 0] = 0
        red [green == 0] = 255
        green [green > 0] = 255
        mask = np.stack([blue, green, red], axis=-1)
        
        _, _, n_clusters = cluster(mask[:, :, 1])

        overlay = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
        
        if self.display_text:
            text = "{} of {} images: {} clusters".format(i + 1, self.total, n_clusters)
            cv2.putText(overlay, text, (25, 25), self.font, 1, (255, 255, 255), 3, cv2.LINE_AA)

        return img, overlay

    def _key_reaction(self):
        '''
        Need uniform reaction for both windows
        '''
        key = cv2.waitKey(1) & 0xFF

        if key == 27: return True
        elif key == ord('d'): self.display_text = not self.display_text
        elif key == ord('n'):
            self.i += 1
            # it it over?
            if self.i >= len(self.im_files):
                return True
            self.im, self.overlay = self.display_image_and_overlay(self.im_files[self.i], self.i)
        return False

    def walk_images(self):
        '''
        Walk the directory and display each image and its overlay
        '''
        cv2.namedWindow(self.im_window)
        cv2.namedWindow(self.mask_window)
        
        self.i = 0
        self.im, self.overlay = self.display_image_and_overlay(self.im_files[self.i], self.i)

        while True:
            cv2.imshow(self.im_window, self.im)
            if self._key_reaction(): break

            cv2.imshow(self.mask_window, self.overlay)
            if self._key_reaction(): break
            