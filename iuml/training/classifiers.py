from __future__ import print_function, division

from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50

from keras.applications import vgg16, inception_v3, xception, resnet50
import keras.applications.imagenet_utils as imagenet_utils

from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import SGD, Adam

import copy

import numpy as np
import abc
from .train import TrainClassifierBase
from .train_inception_base import TrainClassifierInceptionBase

class TrainClassifierResnet50(TrainClassifierBase):
    '''
    Fine-tune resnet models based on the keras_resnet package
    '''

    def __init__(self, root_dir, batch_size=32, weights_file = None, model_file = None, epochs = 5,
                    n_classes = 2, img_shape = (224, 224), class_mode = 'binary'):

        super(TrainClassifierResnet50, self).__init__(root_dir, batch_size, weights_file = weights_file, model_file = model_file,
                            epochs = epochs, n_classes = n_classes, img_shape = img_shape, class_mode = class_mode)

    def configure_model(self, multi_gpu = False):
        '''
        Download & configure the appropriate resnet
        '''
        self.model = ResNet50(weights = 'imagenet', include_top=False, input_shape=self.img_shape + (3,))

        # freeze all layers
        for layer in self.model.layers:
            layer.trainable = True

        units = len(self.n_classes) if self.class_mode == 'categorical' else 1
        
        x = Flatten()(self.model.layers[-1].output)
        predictor = Dense(units=units, activation='sigmoid')(x)
        self.model = Model(self.model.input, predictor)

        self.optimizer = Adam(lr=1e-5) if self.optimizer is None else self.optimizer

        self.model = self.get_parallel_model(self.model) if multi_gpu else self.model
    
    def pretrain(self):
        self.model.compile(optimizer=self.optimizer, loss='{}_crossentropy'.format(self.class_mode), metrics=['accuracy'])

    @property
    def network_name(self):
        return "resnet50"

    @property
    def rescale(self):
        return 1.

    def get_network_module(self):
        return resnet50

    def pretrain(self):
        pass

class TrainClassifierVgg16(TrainClassifierBase):
    """Fine-tule the VGG16 net"""

    def __init__(self, root_dir, batch_size=32, weights_file = None, model_file = None, epochs = 5,
                    n_classes = 2, img_shape = (224, 224), class_mode = 'binary'):

        super(TrainClassifierVgg16, self).__init__(root_dir, batch_size, weights_file = weights_file, model_file = model_file,
                                                        img_shape = img_shape, epochs = epochs, n_classes = n_classes, class_mode = class_mode)

    def configure_model(self, multi_gpu=False):
        '''
        Configure model for training (VGG16)
        '''
        self.vgg16 = VGG16(weights='imagenet')
        fc2 = self.vgg16.get_layer('fc2').output
        prediction = Dense(units=1, activation='sigmoid', name='logit')(fc2)
        self.model = Model(inputs=self.vgg16.input, outputs=prediction)

        for layer in self.model.layers:
            if layer.name in ['fc1', 'fc2', 'logit']:
                continue
            layer.trainable = False

        sgd = SGD(lr=1e-4, momentum=0.9)

        # if we have multiple gpus - train on both
        self.model = self.get_parallel_model(self.model) if multi_gpu else self.model

        self.model.compile(optimizer=sgd, loss='{}_crossentropy'.format(self.class_mode), metrics=['accuracy'])

    @property
    def network_name(self):
        return "vgg16"

    @property
    def rescale(self):
        return 1.

    def get_network_module(self):
        return vgg16

    def pretrain(self):
        pass

class TrainClassifierInceptionV3(TrainClassifierInceptionBase):
    """Fine-tule InceptionV3 net"""

    def __init__(self, root_dir, batch_size=32, epochs=3, weights_file = None, model_file = None,
                    n_classes = 2, img_shape = (299, 299), class_mode='binary'):

        super(TrainClassifierInceptionV3, self).__init__(root_dir, batch_size, epochs, n_classes = n_classes, weights_file = weights_file,
                                                                img_shape = img_shape, model_file = model_file, class_mode=class_mode)

    @property
    def network_name(self):
        return "inception_v3"

    @property
    def Network(self):
        return InceptionV3

    @property
    def train_layer_cutoff(self):
        return 172

    def get_network_module(self):
        return inception_v3

class TrainClassifierXception(TrainClassifierInceptionBase):
    """Fine-tule Xception net"""

    def __init__(self, root_dir, batch_size = 32, epochs = 3, weights_file = None, model_file = None, n_classes = 2, img_shape = (299, 299), class_mode='binary'):
        super(TrainClassifierXception, self).__init__(root_dir, batch_size, epochs, n_classes = n_classes, weights_file = weights_file,
                                                            img_shape = img_shape, model_file = model_file, class_mode=class_mode)

    @property
    def network_name(self):
        return "xception"

    def get_network_module(self):
        return xception

    @property
    def Network(self):
        return Xception

    @property
    def train_layer_cutoff(self):
        return 126
