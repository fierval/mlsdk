from __future__ import print_function, division, absolute_import

from fnmatch import fnmatch
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import multi_gpu_model
from keras.models import load_model

import numpy as np
import os
import abc
import math
import cv2
import iuml.tools.validate as validate
from iuml.tools.image_utils import get_image_files, image_exts
from iuml.tools.net import download_images

from . import utils
from . import predict_generators as pred_gens

def walk_one(dir):
    total = 0
    for root, dirs, files in os.walk(dir):
        if len(files) == 0: continue
        image_files = []
        for ext in image_exts():
            image_files += list(filter(lambda f: fnmatch(f, ext), files))
        total += len(image_files)

    return total

class TrainClassifierBase(object):
    """Wraps fine-tuning of a given network"""

    __metaclass__ = abc.ABCMeta

    training = "training"
    validation = "validation"

    def __init__(self, root_dir, batch_size=32, weights_file = None, model_file = None, n_classes = 2, img_shape = None, epochs=5, class_mode = 'categorical'):
        '''
        Parameters:
            root_dir -- root directory for the dataset of training and validation images
            val_fraction -- fraction used for validation
            batch_size -- batch size
            weights file -- load existing weights
            model_file -- load existing model (weights & model structure)
            epochs -- number of epochs
            class_mode -- 'categorical' or 'binary'
        '''
        self._validation_generator = None
        self._training_generator = None
        self.optimizer = None

        self.custom_objects = None
        self.root_dir = root_dir
        self.model = None
        self.train_dir = os.path.join(root_dir, self.training)
        self.valid_dir = os.path.join(root_dir, self.validation)
        self.model_file = model_file

        self.img_shape = img_shape
        self.n_classes = n_classes
        self.patience = 8

        self._training_examples = None
        self._validation_examples = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_mode = class_mode

        if weights_file is not None and not os.path.exists(weights_file):
            raise ValueError("Weights file does not exist")

        self.weights_file = weights_file

        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        # non-existent directory
        # if it's an empty string - don't validate it.
        # we are just interested in predictions
        if self.root_dir != "":
            validate.raise_if_not_exists(self.root_dir)

    @abc.abstractmethod
    def configure_model(self, multi_gpu = False):
        '''
        Configure model for training.
        Parameters
            multi_gpu - use multiple GPUs if available
        '''
        return

    @abc.abstractproperty
    def network_name(self):
        ''' Name of the network '''
        return

    @property
    def image_size(self):
        ''' Size of the image: (WxH) '''
        return self.img_shape

    def display_model(self):
        '''
        Display model summary
        '''
        self.model.summary()

    @abc.abstractproperty
    def rescale(self):
        return

    @property
    def model_file_name(self):
        return "model_{}.hd5".format(self.network_name)

    @property
    def training_generator(self):
        if self._training_generator is None:
            train_datagen = ImageDataGenerator(preprocessing_function=self.get_preprocess_input(),
                                               rescale = self.rescale,
                                               rotation_range=40,
                                               width_shift_range=0.2,
                                               height_shift_range=0.2,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True,
                                               fill_mode='nearest')

            self._training_generator = train_datagen.flow_from_directory(directory = self.train_dir,
                                                                target_size = self.image_size,
                                                                batch_size = self.batch_size,
                                                                class_mode = self.class_mode)
        return self._training_generator

    @property
    def validation_generator(self):
        if self._validation_generator is None:
            validation_datagen = ImageDataGenerator(preprocessing_function=self.get_preprocess_input(), rescale = self.rescale)

            self._validation_generator = validation_datagen.flow_from_directory(directory = self.valid_dir,
                                                                          target_size = self.image_size,
                                                                          batch_size = self.batch_size,
                                                                          class_mode=self.class_mode)
        return self._validation_generator

    @property
    def training_examples(self):
        if self._training_examples is None:
            self.get_total_examples()
        return self._training_examples

    @property
    def validation_examples(self):
        if self._validation_examples is None:
            self.get_total_examples()

        return self._validation_examples

    @abc.abstractmethod
    def pretrain(self):
        return

    @abc.abstractmethod
    def get_network_module():
        ''' Used to identify the actual keras module that hosts
            an implementation of the network being fine-tuned'''
        return

    def get_preprocess_input(self):
        '''Wrapper around keras.applications.*.preprocess_input()
        to make it compatible for use with keras.preprocessing.image.ImageDataGenerator's
        `preprocessing_function` argument.

        Parameters
        ----------
        x : a numpy 3darray (a single image to be preprocessed)

        Note we cannot pass keras.applications.*.preprocess_input()
        directly to to keras.preprocessing.image.ImageDataGenerator's
        `preprocessing_function` argument because the former expects a
        4D tensor whereas the latter expects a 3D tensor. Hence the
        existence of this wrapper.

        Returns a numpy 3darray (the preprocessed image).

        '''

        def preprocess_input(x):
            module = self.get_network_module()
            return module.preprocess_input(np.expand_dims(x.astype(np.float), axis=0))[0]

        return preprocess_input

    def get_num_gpus(self):
        return utils.get_num_of_gpus()

    def get_parallel_model(self, model):
        '''
        If we have multiple GPUs take advantage of them
        '''

        num_gpus = self.get_num_gpus()
        return multi_gpu_model(model, gpus=num_gpus) if num_gpus > 1 else model

    def train(self, multi_gpu = False, log_dir = 'logs'):
        '''
        Run actual training
        '''

        self.instantiate_model(custom_objects=self.custom_objects)

        print("Starting training for {}, {} samples".format(self.train_dir, self.training_examples))
        print("Validation dataset: {}, {} samples".format(self.valid_dir, self.validation_examples))

        #Create saving callback to store the best model
        self.model_file = os.path.join(self.root_dir, self.model_file_name)
        saving_callback = ModelCheckpoint(self.model_file, monitor='val_loss', verbose=1,
                                    save_best_only=True, save_weights_only=False, period=1)
        
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=self.patience)

        tensorboard_callback = TensorBoard(log_dir=os.path.join(self.root_dir, log_dir), 
                                           write_graph=True, batch_size = self.batch_size, histogram_freq=0)

        return self.model.fit_generator(self.training_generator,
                            steps_per_epoch = self.training_examples // self.batch_size,
                            epochs = self.epochs,
                            callbacks = [saving_callback, tensorboard_callback, early_stopping_callback],
                            validation_data = self.validation_generator,
                            validation_steps = self.validation_examples // self.batch_size)

    def instantiate_model(self, predict=False, custom_objects = None):
        '''
        Creates the underlying model either by building it or loading from a checkpoint
        Parameters:
            predict -- is the model being created for prediction (default: False)
        '''
        if self.model is None and self.weights_file is None and self.model_file is None:
            if predict:
                raise ValueError("For predictions on clean models need to specify model_file")
            else:
                self.configure_model()
                self.pretrain()
        elif self.model is None and self.weights_file is None:
            self.model = load_model(self.model_file, custom_objects = custom_objects)
            print("Loaded model & weights")
        elif self.model is None and self.model_file is None:
            self.configure_model()
            self.pretrain()
            self.model.load_weights(self.weights_file)

    def predict(self, data = None, dir = None, urls=None, capacity = None):
        '''
        Run predictions from a directory or a dataset.
        Parameters:
            data - dataset. If not None - predictions will be run from the dataset, dir is ignored
                    This is just an array-like of image tiles, each of which is HxW[xC]
            dir - should not be None if dataset is None. Runs predictions from the directory
            urls - URLs of images
            out_dir - if specified when urls is not None, save images downloaded from the urls to this directory
        '''

        # enforce that exactly one of the parameters is set
        nones = np.array([1 if d is not None else 0 for d in [data, dir, urls]])
        if nones.sum() == 0 or nones.sum() > 1:
            raise ValueError("One and exactly one of data, urls, dir must be set")

        self.instantiate_model(predict=True)
        prep_input = self.get_preprocess_input()

        input_shape = self.model.input_shape[1:3][::-1]
        if input_shape[0] is None or input_shape[1] is None:
            input_shape = self.img_shape

        if capacity is None: capacity = 100

        gen = pred_gens.get_generator(self.batch_size, input_shape, prep_input, data = data, dir = dir, urls = urls, capacity=capacity)

        preds = self.model.predict_generator(gen, steps=math.ceil(gen.num_images / self.batch_size), verbose = 1)
        return preds[:gen.num_images]

    def split_train_val_data(self, data_root, val_fraction = 0.1):
        '''
        Returns:
            (total_training_images, total_validation_images)
        '''
        (self._training_examples, self._validation_examples) = \
            utils.split_training_validation_data(data_root, self.root_dir, validate_fraction = val_fraction)
        return self._training_examples, self._validation_examples

    def get_total_examples(self):
        self._training_examples, self._validation_examples = walk_one(self.train_dir), walk_one(self.valid_dir)
