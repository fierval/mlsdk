'''
Generates data for RetinaNet based on JSON annotations file
'''


from __future__ import print_function, division, absolute_import

from .generator import Generator
import cv2
import os
import numpy as np
from iuml.tools.annotations import image_descriptors_to_dataframe, points_from_json
from iuml.tools.image_utils import get_image_files
from numba import jit
import six
from .utils import image as util_image

TRAINING = 1
VALIDATION = 2

import warnings
warnings.simplefilter('ignore')

class AnnotationsGenerator(Generator):
    '''
    Read the annotations file and feed images for training
    '''
    def __init__(self, json_annotations_file, class_map, base_dir = None, image_shape = (512, 512), type = TRAINING, **kwargs):
        '''
        Parameters:
            json_annotations_file - iUML annotator produced annotations
            class_map - dictionary of {id: name} that maps class ids to names. These names are assumed to be subdirectories of the base_dir
            base_dir - root directory for training (or validation) images
            type - indicating whether this instance is being used for training (in which case "training" is appended to base_dir).
                    or "validation" in which case "validation" is appended.
        '''
        self.image_names = []
        self.image_data  = {}
        self.base_dir    = base_dir
        self.halftile = None
        self.labels = class_map

        if type == TRAINING:
            kwargs['group_method'] = 'random'
            kwargs['shuffle_groups'] = True
        else:
            kwargs['group_method'] = None
            kwargs['shuffle_groups'] = False

        if 'batch_size' not in kwargs.keys():
            kwargs['batch_size'] = 1

        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            self.base_dir = os.path.dirname(json_annotations_file)

        if not os.path.exists(self.base_dir):
            raise FileExistsError("Directory {} does not exist".format(self.base_dir))

        self.image_data = points_from_json(json_annotations_file)

        # part of the images are under the 'validation' tree
        # and part - under 'training'
        self.image_data = self.filter_image_data_by_directory()

        self.image_names = list(self.image_data.keys())

        image_file = os.path.join(self.base_dir, self.image_names[0])
        im = cv2.imread(image_file)
 
        if image_shape is not None:
            image_min_side, image_max_side = min(image_shape), max(image_shape)
        else:
            imshape = im.shape[:2]
            image_min_side, image_max_side = min(imshape), max(imshape)

        kwargs['image_min_side'] = image_min_side
        kwargs['image_max_side'] = image_max_side
        
        if 'tile' in kwargs.keys():
            self.halftile = kwargs['tile'] // 2
            # resize the tile if we have to
    
            _, scale = util_image.resize_image(im, image_min_side, image_max_side)
            self.halftile *= scale
            del kwargs['tile']
        # reverse of class map
        self.classes = {v: k for k, v in six.iteritems(self.labels)}

        super(AnnotationsGenerator, self).__init__(**kwargs)

    def filter_image_data_by_directory(self):
        '''
        Removes annotations not in the current dataset
        '''
        files = {os.path.split(f)[1] for f in get_image_files(self.base_dir)}
        
        all_names = set(self.image_data.keys())

        return {k: v for k, v in six.iteritems(self.image_data) if k in files}
        
    def size(self):
        return len(self.image_names)

    def num_classes(self):
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def image_path(self, image_index):
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        image = cv2.imread(self.image_path(image_index))
        return float(image.shape[1]) / float(image.shape[0])

    def load_image(self, image_index):
        return cv2.cvtColor(cv2.imread(self.image_path(image_index)), cv2.COLOR_BGR2RGB)

    @jit
    def load_annotations(self, image_index):
        path   = self.image_names[image_index]
        annots = self.image_data[path]
        boxes  = np.zeros((len(annots), 5))

        for idx, annot in enumerate(annots):
            boxes[idx, 4] = float(annot['class'])
        
            x1, y1, x2, y2 = self.get_annotation_rect(annot)    
            boxes[idx, 0] = float(x1)
            boxes[idx, 1] = float(y1)
            boxes[idx, 2] = float(x2)
            boxes[idx, 3] = float(y2)

        return boxes

    @jit
    def get_annotation_rect(self, annot):
        '''
        Given an annotation dictionary, return the left, top, right, bottom rectangle
        '''
        
        if self.halftile is not None:
            delta = self.halftile
        elif 'r' in annot.keys():
            delta = annot['r']
        else:
            delta = None
    
        if delta is not None:
            x1 = float(max(0, annot['x'] - delta))
            x2 = float(annot['x'] + delta)
            y1 = float(max(0, annot['y'] - delta))
            y2 = float(annot['y'] + delta)
            
        else:
            x1 = float(annot['x'])
            x2 = float(annot['x2'])
            y1 = float(annot['y'])
            y2 = float(annot['y2'])
            
        return x1, y1, x2, y2