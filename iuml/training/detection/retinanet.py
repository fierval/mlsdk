from __future__ import print_function, division, absolute_import

import numpy as np
from iuml.training import TrainClassifierBase

from .model import *
from .losses import *

import keras_resnet
from keras_resnet.models import ResNet50, ResNet101, ResNet152

from keras.layers import Input
from keras.optimizers import Adam

from iuml.training.utils import freeze_batch_normalization_layers, clean_training_validation_trees
from sklearn.model_selection import train_test_split
from .model import custom_objects as custom_objects_local
from .jsongen import AnnotationsGenerator, TRAINING, VALIDATION
from iuml.tools.image_utils import get_image_files
from iuml.tools.validate import create_if_not_exists
from .utils import image as util_image
import cv2

import warnings
warnings.simplefilter('ignore', DeprecationWarning)

import shutil
import os

resnet_filename = 'ResNet-{}-model.keras.h5'
resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)

custom_objects = custom_objects_local.copy()
custom_objects.update(keras_resnet.custom_objects)

allowed_backbones = ['resnet50', 'resnet101', 'resnet152']


def download_imagenet(backbone):
    validate_backbone(backbone)

    backbone = int(backbone.replace('resnet', ''))

    filename = resnet_filename.format(backbone)
    resource = resnet_resource.format(backbone)
    if backbone == 50:
        checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    elif backbone == 101:
        checksum = '05dc86924389e5b401a9ea0348a3213c'
    elif backbone == 152:
        checksum = '6ee11ef2b135592f8031058820bb9e71'

    return keras.applications.imagenet_utils.get_file(
        filename,
        resource,
        cache_subdir='models',
        md5_hash=checksum
    )


def validate_backbone(backbone):
    if backbone not in allowed_backbones:
        raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

class RetinaNet(TrainClassifierBase):

    def __init__(self, root_dir, json_annotations_file, class_map, backbone = 'resnet50', batch_size=2, weights_file = None, model_file = None, epochs = 5,
                    img_shape = None, tile = None):

        '''
        Instantiate RetinaNet
        '''

        self.backbone = backbone
        validate_backbone(self.backbone)

        self.tile = tile
        self.json_annotations_file = json_annotations_file
        self.class_map = class_map
        n_classes = max(list(class_map.keys())) + 1

        super(RetinaNet, self).__init__(root_dir, batch_size, weights_file = weights_file, model_file = model_file,
                            epochs = epochs, n_classes = n_classes, img_shape = img_shape)
        
        self.custom_objects = custom_objects

    def split_train_val_data(self, data_root, val_fraction = 0.1):
        '''
        Returns:
            (total_training_images, total_validation_images)
        '''

        image_files = get_image_files(data_root)

        train_images, test_images = train_test_split(image_files, test_size=val_fraction)

        # create directories if they don't exist
        all_dirs = []

        clean_training_validation_trees(self.train_dir, self.valid_dir)

        for dir in [self.train_dir, self.valid_dir]:
            all_dirs.append(dir)
            create_if_not_exists(dir)

        all_groups = [train_images, test_images]

        # copy files into their training/validation directories
        print('Splitting into directories...')
        for dir, im_files in zip(all_dirs, all_groups):
            print('Copying to: {}'.format(dir))
            for src in im_files:
                fn = os.path.split(src)[1]
                dest = os.path.join(dir, fn)
                shutil.copy(src, dest)

        self._training_examples = len(train_images)
        self._validation_examples = len(test_images)

        return self._training_examples, self._validation_examples

    def create_model(self, num_classes, backbone='resnet50', inputs=None, **kwargs):
        '''
        Create the underlying ResNet backbone
        '''
        validate_backbone(backbone)

        # choose default input
        if inputs is None:
            inputs = keras.layers.Input(shape=(None, None, 3))

        # create the resnet backbone
        if backbone == 'resnet50':
            resnet = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)
        elif backbone == 'resnet101':
            resnet = keras_resnet.models.ResNet101(inputs, include_top=False, freeze_bn=True)
        elif backbone == 'resnet152':
            resnet = keras_resnet.models.ResNet152(inputs, include_top=False, freeze_bn=True)

        # create the full model
        model = retinanet_bbox(inputs=inputs, num_classes=num_classes, backbone=resnet, **kwargs)

        return model
    
    def configure_model(self, multi_gpu = False):
        
        print("Creating RetinaNet model...")
        inputs = Input(shape=(None, None, 3))
        # TODO: otherwise we instantiate with weights
        if self.weights_file is None:
            weights = download_imagenet(self.backbone)
            print("Loaded weights")

        # configure resnet with non-maximum suppression
        print("Configuring model...")
        self.model = self.create_model(self.n_classes, self.backbone, inputs=inputs, nms=True)
        self.model.load_weights(weights, by_name=True, skip_mismatch=True)

        # compile model
        self.model.compile(
            loss={
                'regression'    : smooth_l1(),
                'classification': focal()
            },
            optimizer=Adam(lr=1e-5, clipnorm=0.001)
        )
        print("Model creation...Done")

    def get_generator(self, generator_kind):
        '''
        Returns generators
        Parameters:
            generator_kind - 'training', 'validation'
        '''
        if generator_kind == 'training':
            base_dir = self.train_dir
            type = TRAINING
        elif generator_kind == 'validation':
            base_dir = self.valid_dir
            type = VALIDATION
        else:
            raise NotImplementedError("Unknown kind of generator")

        return AnnotationsGenerator(self.json_annotations_file, self.class_map, base_dir = base_dir, type = type,
                                    tile = self.tile, image_shape = self.image_size)

    @property
    def training_generator(self):
        if self._training_generator is None:
            self._training_generator = self.get_generator('training')
        return self._training_generator

    @property
    def validation_generator(self):
        if self._validation_generator is None:
            self._validation_generator = self.get_generator('validation')
        return self._validation_generator

    @property
    def network_name(self):
        return "retinanet"

    def predict(self, data = None, dir = None):
        '''
        Run predictions from a directory or a dataset.
        Parameters:
            data - dataset. If not None - predictions will be run from the dataset, dir is ignored
                    This is just an array-like of image tiles, each of which is HxW[xC]
            dir - should not be None if dataset is None. Runs predictions from the directory
        '''

        self.instantiate_model(predict=True, custom_objects=custom_objects)
        
        if data is None:
            im_files = get_image_files(dir)
            data = [cv2.imread(f) for f in get_image_files(dir)]

        data = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in data]

        if self.image_size is not None:
            min_side, max_side = min(self.image_size), max(self.image_size)
            data = [np.expand_dims(util_image.resize_image(im, min_side, max_side)[0], axis=0) for im in data]
        else:
            data = [np.expand_dims(im, axis=0) for im in data]

        data = [keras.applications.imagenet_utils.preprocess_input(im.astype(np.float32)) for im in data]
        data = np.vstack(data)

        return self.model.predict(data, batch_size = self.batch_size)