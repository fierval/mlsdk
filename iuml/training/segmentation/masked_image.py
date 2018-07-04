from __future__ import print_function, division

import os
import cv2
import random
import numpy as np
import glob
from ...tools import validate
from ...tools.image_utils import get_image_files
import copy
import math

from keras.preprocessing import image

def get_file_name(f):
    return os.path.splitext(os.path.split(f)[1])[0]

class MaskedImageDataGenerator(image.Iterator):
    '''
    Data generator for the unet.
    '''

    def __init__(self, images_folder, masks_folder, batch_size = 1, image_shape = (512, 512), preprocess_input = None, shuffle = True, seed = 0, n_classes = 2, augment = True):
        '''
        Parameters:
            images_folder - actual images
            masks_folder - ground truth masks: images of type CV_8U1 or CV_8U3 where each pixel = value of class (0, 1,...)
            batch_size - number of images in one batch
            image_shape - target shape of an image (cols, rows)
            preprocess_input - function that takes a 3 channel image and normalizes it
            shuffle - shuffle between epochs
            seed - random seed
            n_classes - how many total classes we have (cannot be less than 2)
        '''

        if batch_size < 1:
            raise ValueError("Batch size should be >= 1")

        if n_classes < 2:
            raise ValueError("Too few classes (at least 2)")

        validate.raise_if_not_exists(images_folder)
        validate.raise_if_not_exists(masks_folder)

        # dimensions as (row, col)
        self.image_shape = image_shape
        self.target_size = image_shape[::-1]
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.augment = augment
        self.image_files = get_image_files(images_folder)
        self.mask_files = glob.glob(os.path.join(masks_folder, '*.npz'))

        self.image_files.sort(key=get_file_name)
        self.mask_files.sort(key=get_file_name)

        self.preprocess_input =  preprocess_input
        print("sorted datasets...")

        if len(self.mask_files) != len(self.image_files):
            raise ValueError("Images and masks directories contain different number of files")


        super(MaskedImageDataGenerator, self).__init__(len(self.image_files), self.batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        '''
        Return a batch of images.
        '''
        batch_image = np.zeros((len(index_array),) + self.target_size + (3,), dtype=np.float)
        batch_mask = np.zeros((len(index_array),) + self.target_size + (self.n_classes,), dtype=np.float)

        for i, idx in enumerate(index_array):
            img = cv2.imread(self.image_files[idx])
            mask = np.load(self.mask_files[idx])['labels']

            img_name = get_file_name(self.image_files[idx])
            mask_name =  get_file_name(self.mask_files[idx])

            if (img_name != mask_name):
                raise ValueError("{} != {}".format(img_name, mask_name))

            batch_image[i], batch_mask[i] = self.augmentation(img, mask)
            if self.preprocess_input is not None:
                batch_image[i] = self.preprocess_input(batch_image[i])
        return batch_image, batch_mask

    def next(self):
        '''
        Advancing iterator
        '''
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

    def rotrand(self, image, mask):

        randrot = random.randint(0,3)

        image = np.rot90(image, k=randrot)
        mask = np.rot90(mask, k=randrot)

        return image, mask

    def linear(self, m_max = 0.1, b_max = 20):

        rand = random.random() * m_max
        rand = rand * 2 - rand
        mrand = 1 + rand
        brand = random.randint(-1 * b_max, b_max)
        return mrand, brand

    # TODO: Since masks are being pre-processed
    # into the Unet shape (x, y, 2) - not sure
    # if it's worth it to apply this due to the
    # immenent resizing using cv2.
    # also linear interpolation of pixel values
    # during resizing may not be the best idea
    def randzoom(self, image, mask, maxzoom=0.15):

        minzoomcoord = maxzoom
        maxzoomcoord = 1-maxzoom

        randx1 = random.randint(0,int(self.target_size[0]*minzoomcoord))
        randx2 = random.randint(int(maxzoomcoord*self.target_size[0]) , self.target_size[0])

        randy1 = random.randint(0,int(self.target_size[1]*minzoomcoord))
        randy2 = random.randint(int(maxzoomcoord*self.target_size[1]) , self.target_size[1])

        # elipses rather than a column.
        # Means we don't care about whatever dimensions follow.
        # There may or may not be more...
        mask = mask[randx1:randx2, randy1:randy2, ...]
        image = image[randx1:randx2, randy1:randy2, ...]

        return image, mask

    def mult_random(self, image, mask, m_max=0.1, b_max = 20):
        mrand, brand = self.linear(m_max, b_max)

        def multadd(c):
            return mrand * c + brand

        image[:,:,0] = multadd(image[:,:,0])
        image[:,:,1] = multadd(image[:,:,1])
        image[:,:,2] = multadd(image[:,:,2])
        return image, mask

    def augmentation(self, imgdata, mskdata):

        if self.augment:
            process_chance = random.randint(0,1)

            if process_chance==1:

                choices = [self.rotrand, self.mult_random]
                choicenum = random.randint(1,len(choices))

                for i in range(choicenum):

                    choice = choices.pop(random.randint(0,len(choices)-1))
                    imgdata, mskdata = choice(imgdata, mskdata)

        if imgdata.shape[:2][::-1] != self.image_shape:
            imgdata = cv2.resize(imgdata, self.image_shape, interpolation=cv2.INTER_LINEAR)
            mskdata = cv2.resize(mskdata, self.image_shape, interpolation=cv2.INTER_LINEAR)

        return (imgdata, mskdata)