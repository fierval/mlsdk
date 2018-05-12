from __future__ import print_function, division, absolute_import

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD

import numpy as np
import abc
from .train import TrainClassifierBase

class TrainClassifierInceptionBase(TrainClassifierBase):
    '''
    Base class for Inception derivied classifiers
    '''
    __metaclass__ = abc.ABCMeta
    def __init__(self, root_dir, batch_size=32, epochs=3, weights_file = None, model_file = None, n_classes = 2, img_shape = (224, 224), class_mode='binary'):
        super(TrainClassifierInceptionBase, self).__init__(root_dir, batch_size, weights_file = weights_file, n_classes = n_classes,
                                                                img_shape = img_shape, class_mode=class_mode, model_file = model_file,  epochs = epochs)

    @abc.abstractproperty
    def Network(self):
        return

    @abc.abstractproperty
    def train_layer_cutoff(self):
        return

    @property
    def rescale(self):
        return 1./255.

    @property
    def image_size(self):
        return [299, 299]

    def configure_model(self, multi_gpu = False):
        '''
        Configure model for training
        '''
        print("Getting {} weights...".format(self.network_name))
        self.base_model = self.Network(weights='imagenet', include_top=False)

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)

        # trainable FC layers
        x = Dense(1024, activation='relu')(x)
        # trainable classification layer
        predictions = Dense(units=1, activation='sigmoid')(x)

        # combine model
        self.model = Model(inputs=self.base_model.input, outputs=predictions)
        self.multi_gpu = multi_gpu

    def pretrain(self):
        if self.weights_file is None:
            # Freeze all base model layers
            for layer in self.base_model.layers:
                layer.trainable = False

            # compile initial model
            self.model.compile(optimizer='rmsprop', loss='{}_crossentropy'.format(self.class_mode), metrics=['accuracy'])

            print("Starting pre-training: training top layer on existing weights")
            self.model.fit_generator(self.training_generator,
                                steps_per_epoch = self.training_examples // self.batch_size,
                                epochs = self.epochs,
                                validation_data=self.validation_generator,
                                validation_steps = self.validation_examples // self.batch_size)


            print("Unfreezing and training CNN...")
            for layer in self.model.layers[:self.train_layer_cutoff]:
                layer.trainable = False

            for layer in self.model.layers[self.train_layer_cutoff:]:
                layer.trainable = True

            # if we have multiple gpus - train on both
            self.model = self.get_parallel_model(self.model) if self.multi_gpu else self.model

        print("Compiling model...")
        self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='{}_crossentropy'.format(self.class_mode), metrics=['accuracy'])
