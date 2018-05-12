from __future__ import division, print_function

import json
import os
import fnmatch
import cv2
import numpy as np
import copy
import math
import multiprocessing as mp
from iuml.tools.clusters import *
import random
import time

class Recorder(object):
    '''
    Record a video from image, mask & fps
    '''

    def __init__(self, image_file, mask_file, video_file, fps = 30., scale = 0.5, initial = 8, q = 2.):
        '''
        Parameters:
            image_file - original image file
            mask_file - extracted mask
            video_file - output video
            fps - desired frames per sec (default: 30)
            scale - scale the image by... (default: 1)
            initial - inital buds to display (default: 8)
            q - geometrical progression coefficient (default: 2)
        '''
        if not 0 < scale <= 1:
            raise ValueError("Scale should be between postive less than 1")

        if not os.path.exists(image_file):
            raise IOError("image file does not exist")

        if not os.path.exists(mask_file):
            raise IOError("mask does not exist")

        if os.path.exists(video_file):
            os.remove(video_file)
        
        self.window_title = "Bud Finder"
        self.image = cv2.imread(image_file)
        if scale != 1:
            self.image = cv2.resize(self.image, None, fx=scale, fy=scale)
            self.mask = cv2.resize(self.mask, None, fx=scale, fy=scale)

        self.mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        self.fps = fps
        self.initial = initial
        self.q = q
                
        if self.fps <= 0 or self.fps > 60:
            raise ValueError("fps out of range: should be 0 < fps <= 60")

        if self.image is None:
            raise IOError("Could not read image")
        
        if self.mask is None:
            raise IOError("Could not read mask")
        
        # DIVX is a supported codec on Windows
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.video = cv2.VideoWriter(video_file, fourcc, int(fps), self.image.shape[::-1][1:])
        
        # seed random generator
        random.seed(os.urandom(10))

        self.mask = cv2.GaussianBlur(self.mask, (5, 5), 0)
        _, self.mask = cv2.threshold(self.mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # set it up: get clusters and their centers
        X, labels, n_clusters = cluster(self.mask)

        # need to be able to index it via lists, so transform to numpy array
        self.boxes = np.array(map_clusters_to_bounding_boxes(labels, X, n_clusters))

        self.centers = list(map(get_box_center, self.boxes))
        
    def animate_display(self, idxs):
        '''
        Display animated circles
        Returns: True - continue display False - stop
        '''

        fps = self.fps # shorthand

        img = self.image.copy()
        draw_clusters(img, self.boxes[idxs], radius=12, color=(0,255,0))

        step = 1. / fps
        alpha = step

        while alpha <= 1:
            display_image = cv2.addWeighted(self.image, (1-alpha), img, alpha, 0)
            self.video.write(display_image)

            cv2.imshow(self.window_title, display_image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: #ESC
                return False
#            time.sleep(4 * step)
            
            alpha += step / 4
        
        self.image = img
        return True

    def record(self, all_at_once=False):
        
        # indices into X from which we'll sample
        idxs = set(range(len(self.centers)))
        n_disp_buds = self.initial if not all_at_once else len(idxs)
        q = self.q

        cv2.namedWindow(self.window_title)
        dontstop = True

        while len(idxs) > 0 and dontstop:
            sample_size = int(n_disp_buds) if int(n_disp_buds) < len(idxs) else len(idxs)

            to_display = random.sample(idxs, sample_size)
            idxs.difference_update(to_display)
            
            n_disp_buds *= q

            dontstop = self.animate_display(to_display)
            if not dontstop: break

        print("Done!")        
        while True:
            cv2.imshow(self.window_title, self.image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break

        cv2.destroyAllWindows()
        self.video.release()
        