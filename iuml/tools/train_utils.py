import copy
from iuml.training import *
from iuml.training.segmentation import *
from iuml.training.detection import *

def create_trainer(net, root_train, **kwargs):
    '''
    Creates a trainer based on kwargs:
        batch_size = 32, 
        weights_file = None, 
        epochs = 5, 
        class_mode = 'binary',
    For Unet:
        images = subdirectory of images
        masks = subdirectory of masks
        img_shape = (width, height) of the image
    '''
    
    allowed_trainers = {'Resnet50', 'VGG16', 'InceptionV3', 'Xception', 'Unet', 'RetinaNet'}
    if net not in allowed_trainers:
        raise ValueError("Model not recognized: {}".format(net))
    
    params = kwargs
    if net == 'RetinaNet':
        params = copy.deepcopy(kwargs)
        json_annotations_file = params['json_annotations_file']
        class_map = params['class_map']

        del params['json_annotations_file']
        del params['class_map']

        trainer = RetinaNet(root_train, json_annotations_file, class_map, **params)
    if net == 'Resnet50':
        trainer = TrainClassifierResnet50(root_train, **params)
    elif net == 'VGG16':
        trainer = TrainClassifierVgg16(root_train, **params)
    elif net == 'InceptionV3':
        trainer = TrainClassifierInceptionV3(root_train, **params)
    elif net == 'Xception':
        trainer = TrainClassifierXception(root_train, **params)
    elif net == 'Unet':
        params = copy.deepcopy(kwargs)

        # if we are interested in training - we need images & masks
        # if we are just predicting - they are not necessary
        if 'images'in params.keys():
            images = params['images']
            masks = params['masks']

            del params['images']
            del params['masks']
        else:
            images, masks = "", ""

        if 'class_mode' in params.keys():
            del params['class_mode']

        trainer = Unet(root_train, images, masks, **params)

    return trainer

