from __future__ import print_function, division
import shutil
import glob

import keras.applications.imagenet_utils as keras_imagenet

from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Activation, Concatenate, BatchNormalization
from keras.optimizers import *

import os
import numpy as np
from ..train import TrainClassifierBase
from .masked_image import MaskedImageDataGenerator
from ...tools import validate
from ...tools.image_utils import get_image_files
from sklearn.model_selection import train_test_split

from ..utils import clean_training_validation_trees, create_preprocessing_config, load_normalization_dictionary
from .utils import *
import math

class Unet(TrainClassifierBase):


    def __init__(self, root_folder, images_subfolder, masks_subfolder,
                    batch_size = 1, weights_file = None,
                    preprocess_input = False,
                    model_file = None,
                    img_shape = (512, 512), n_classes = 2, epochs = 5, class_balance = True,
                    loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = None):
        '''
        Dataset structure:
        root_folder
        |
        ---- training
             |
             ----images
             |
             ----masks
        |
        ---- validation
             |
             ----images
             |
             ----masks

        Parameters:
            root_folder - training root
            images_subfolder - subfolder of the root_folder where images are stored
            masks_subfolder - subfolder where masks are stored
            val_fraction - fraction set aside for validation
            weights_file - if we already have a trained model
            preprocess_input - should we use training set-wise preprocessing:
                                center & normalize all of the training set data
                                if set to False - batch normalization will be used
            img_shape - tuple (img_width, img_height) - target image size
            n_classes - number of classes
            epochs - number of epochs to run training
            class_balance - balance the loss by introducing class-balancing factor and down-weighting easy-example
            loss - loss functin. categorical_crossentropy by default
            metrics - metrics to report. Accuracy by default
        '''
        # REVIEW: There is no parameter validation because the object may sometimes
        # be created for the sake of predictions only.

        self.images_folder = images_subfolder
        self.masks_folder = masks_subfolder
        self.n_classes = n_classes
        self.model = None
        self.should_preprocess = preprocess_input
        self.normalization_dict = None

        # in case we want custom loss/metrics
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.class_balance = class_balance

        # suppress model printing statements
        self.verbose = False

        if self.n_classes < 2:
            raise ValueError("Minimum 2 classes required")

        super(Unet, self).__init__(root_folder, batch_size, weights_file=weights_file, model_file = model_file, n_classes = n_classes, img_shape = img_shape, epochs = epochs)

        self.preprocessing_file = os.path.join(self.train_dir, "preprocessing.json")

    def split_train_val_data(self, data_root, val_fraction = 0.1):
        '''
        Returns:
            (total_training_images, total_validation_images)
        '''

        image_files = get_image_files(os.path.join(data_root, self.images_folder))
        mask_files = get_image_files(os.path.join(data_root, self.masks_folder))


        # Verify that files match
        im_files = {os.path.split(f)[1] for f in image_files}
        m_files = {os.path.split(f)[1] for f in mask_files}

        # take symmetric difference. Shold be 0
        if len(im_files ^ m_files) != 0:
            raise ValueError('Masks files and images files do not match')

        train_images, test_images, train_masks, test_masks = train_test_split(image_files, mask_files, test_size=val_fraction)

        # create directories if they don't exist
        # arrange the 4 directories in order train-ims, train-masks, val-ims, val-masks
        all_dirs = []

        clean_training_validation_trees(self.train_dir, self.valid_dir)

        if os.path.exists(self.preprocessing_file):
            os.remove(self.preprocessing_file)

        for dir in [self.train_dir, self.valid_dir]:
            for sub_dir in [self.images_folder, self.masks_folder]:
                d = os.path.join(dir, sub_dir)
                all_dirs.append(d)
                validate.create_if_not_exists(d)

        all_groups = [train_images, train_masks, test_images, test_masks]

        # copy files into their training/validation directories
        print('Splitting into directories...')
        for dir, im_files, is_mask in zip(all_dirs, all_groups, [False, True, False, True]):
            print('Copying to: {}'.format(dir))
            if not is_mask:
                for src in im_files:
                    fn = os.path.split(src)[1]
                    dest = os.path.join(dir, fn)
                    copy_with_resize(src, dest, self.img_shape)
            else:
                for src in im_files:
                    fn = os.path.splitext(os.path.split(src)[1])[0]
                    dest = os.path.join(dir, fn)
                    mask_from_image_to_unet_label(src, dest, self.img_shape, self.n_classes)

        self._training_examples = len(train_images)
        self._validation_examples = len(test_images)

        # we will compute means and averages of the training set
        # and store them for future use
        if self.should_preprocess:
            self.normalization_dict = \
                create_preprocessing_config(self.preprocessing_file, os.path.join(self.train_dir, self.images_folder))

        return self._training_examples, self._validation_examples

    def get_total_examples(self):
        def total_files_in_dir(dir):
            return len(os.listdir(os.path.join(dir, self.images_folder)))

        self._training_examples = total_files_in_dir(self.train_dir)
        self._validation_examples = total_files_in_dir(self.valid_dir)

    def configure_model(self, multi_gpu=False):

        concat = Concatenate(axis=3)

        inputs = Input(self.image_size[::-1] + (3,))
        if self.verbose:
            print ("inputs shape:", inputs.shape)

        # if we are doing our own preprocessing - no need to batch-normalize
        if self.should_preprocess:
            print('abandoning batch normalization')
            if self.normalization_dict is None:
                self.normalization_dict = load_normalization_dictionary(self.preprocessing_file)
                print(self.normalization_dict)
            x = inputs
        else:
            x = BatchNormalization(axis=3,momentum=0.985)(inputs)

        conv1 = Conv2D(64, 3, dilation_rate=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        if self.verbose:
            print ("pool1 shape:", pool1.shape)

        conv2 = Conv2D(128, 3, dilation_rate=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        if self.verbose:
           print ("pool2 shape:", pool2.shape)

        conv3 = Conv2D(256, 3, dilation_rate=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        if self.verbose:
            print ("pool3 shape:", pool3.shape)

        conv4 = Conv2D(512, 3, dilation_rate=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.51)(conv4)
        if self.verbose:
            print ("pool4 shape:", drop4.shape)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4))
        merge7 = concat([conv3,up7])
        conv7 = Conv2D(256, 3, dilation_rate=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        if self.verbose:
            print ("conv7 shape:", conv7.shape)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concat([conv2,up8])
        conv8 = Conv2D(128, 3, dilation_rate=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        if self.verbose:
            print ("conv8 shape:", conv8.shape)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concat([conv1,up9])
        conv9 = Conv2D(64, 3, dilation_rate=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(self.n_classes, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

        if self.verbose:
            print ("conv9 shape:", conv9.shape)

        activation = Activation('softmax')(conv9)

        self.model = Model(inputs = inputs, outputs = activation)

        # if we have multiple gpus - train on both
        self.model = self.get_parallel_model(self.model) if multi_gpu else self.model
        optimizer = Adam(lr=1e-5) if self.optimizer is None else self.optimizer
        self.model.compile(optimizer = optimizer, loss = self.loss, metrics = self.metrics)

        return self.model

    def get_generator(self, generator_kind):
        '''
        Returns generators
        Parameters:
            generator_kind - 'training', 'validation'
        '''
        if generator_kind == 'training':
            ims_folder = os.path.join(self.train_dir, self.images_folder)
            masks_folder = os.path.join(self.train_dir, self.masks_folder)
            return MaskedImageDataGenerator(ims_folder, masks_folder, self.batch_size, self.img_shape, preprocess_input = self.get_preprocess_input(),
                                            shuffle=True, n_classes=self.n_classes, augment=True)

        elif generator_kind == 'validation':
            ims_folder = os.path.join(self.valid_dir, self.images_folder)
            masks_folder = os.path.join(self.valid_dir, self.masks_folder)
            return MaskedImageDataGenerator(ims_folder, masks_folder, self.batch_size, self.img_shape, preprocess_input = self.get_preprocess_input(),
                                            shuffle=False, n_classes=self.n_classes, augment=False)

        else:
            raise NotImplementedError("Unknown kind of generator")

    def predict(self, data = None, dir = None, urls=None, capacity = None):
        preds = super().predict(data, dir, urls, capacity)
        return unet_proba_to_class_masks(preds)

    @property
    def training_generator(self):
        self._training_generator = self.get_generator('training')
        return self._training_generator

    @property
    def validation_generator(self):
        self._validation_generator = self.get_generator('validation')
        return self._validation_generator

    @property
    def rescale(self):
        return 1./255

    @property
    def network_name(self):
        return "unet"

    def get_network_module(self):
        raise NotImplementedError("Not implemented")

    def pretrain(self):
        pass

    def get_preprocess_input(self):
        '''
        Preprocess images
        '''

        def prep_input(x):
            im = x.astype(np.float32)

            # we are centering & normalizing
            im[:, :, 0] -= self.normalization_dict['means'][0]
            im[:, :, 1] -= self.normalization_dict['means'][1]
            im[:, :, 2] -= self.normalization_dict['means'][2]

            im[:, :, 0] /= self.normalization_dict['stds'][0]
            im[:, :, 1] /= self.normalization_dict['stds'][1]
            im[:, :, 2] /= self.normalization_dict['stds'][2]

            return im

        if self.should_preprocess:
            if self.normalization_dict is None:
                self.normalization_dict = load_normalization_dictionary(self.preprocessing_file)
            return prep_input
        else:
            return (lambda x: x.astype(np.float32))