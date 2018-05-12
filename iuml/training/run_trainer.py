from __future__ import print_function, division, absolute_import

import os
import argparse
import iuml.tools.validate as validate

if __name__ == '__main__':

    def parse_args():
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('--root-tiles', action="store",
            help='root directory under which image tiles are stored. If specified data will be dropped under training root and split into training/validation datasets')
    
        arg_parser.add_argument('--root-train', action="store", help='training root directory')
        arg_parser.add_argument('--val-fraction', action='store', type=float, default=0.1,
                                help='between 0 and 1.0: validation fraction. Default: 0.1')
        arg_parser.add_argument('--batch-size', action='store', help='training batch size. Default: 32', type=int, default=32)
        arg_parser.add_argument('--class-map', action='store', help='class map for RetinaNet: {id: \'name\'}')
        arg_parser.add_argument('--epochs', action='store', help='number of epochs', type=int, default=5)

        arg_parser.add_argument('--class-mode', action="store", default='binary', help='binary or categorical: classifier output (default: binary)')
        arg_parser.add_argument('--num-classes', action="store", default=2, type=int, help='Number of classes for Unet (default: 2)')

        arg_parser.add_argument('--net', action='store', default='VGG16',
                                help='Net to fine-tune (InceptionV3, Xception, VGG16, Unet). Default: VGG16')
        arg_parser.add_argument('--images', action='store', help='if net == Unet - name of the "images" subfolder')
        arg_parser.add_argument('--masks', action='store', help='if net == Unet - name of the "masks" subfolder')
        arg_parser.add_argument('--model-weights', action='store', help='name of the file to load weights from', default=None)
        arg_parser.add_argument('--multi-gpu', action='store_true', help='use multiple GPUs if available')
        arg_parser.add_argument('--target-width', action='store', help='target image width (for Unet)', type=int, default=0)
        arg_parser.add_argument('--target-height', action='store', help='target image height (for Unet)', type=int, default=0)
        arg_parser.add_argument('--annotations', action='store', help='annotations file produced by iUNU Image Annotator')
        arg_parser.add_argument('--tile', action='store', help='tile size for object detection', type=int)
        results = arg_parser.parse_args()
        return results, arg_parser

    def main():
        results, arg_parser = parse_args()
        net = results.net
        weights_file = results.model_weights
        multi_gpu = results.multi_gpu
        class_mode = results.class_mode
        batch_size = results.batch_size
        root_train = results.root_train
        annotations_file = results.annotations

        if net == 'RetinaNet':
            class_map = eval(results.class_map)
        else:
            class_map = None

        if weights_file is not None and not os.path.exists(weights_file):
            raise ValueError("Weights file does not exist")

        if root_train is None:
            arg_parser.print_usage()
            return

        val_fraction = results.val_fraction
        if not 0 < val_fraction < 1.0:
            raise ValueError("Validation fraction is between 0 and 1.0")

        epochs = results.epochs
        if epochs < 0:
            raise ValueError('Number of epochs should be positive')

        from iuml.tools.train_utils import create_trainer

        # shorthand for parameters
        params = dict(batch_size=batch_size, weights_file = weights_file, epochs=epochs, class_mode=class_mode)
        params['n_classes'] = results.num_classes

        params['img_shape'] = None if results.target_width == 0 else (results.target_width, results.target_height)

        if class_map is not None:
            params['class_map'] = class_map
            params['json_annotations_file'] = annotations_file
            params['tile'] = results.tile

        if net == 'Unet':
            if validate.is_str_none_or_empty(results.images) or validate.is_str_none_or_empty(results.masks):
                raise ValueError('images or masks folders not specified')

            params['images'] = results.images
            params['masks'] = results.masks
        
        trainer = create_trainer(net, root_train, **params)

        # do we need to create our train/validation data sets?
        root_tiles = results.root_tiles
        if root_tiles != "" and root_tiles is not None:
            if not os.path.exists(root_tiles):
                raise FileExistsError("Directory does not exist: {}".format(root_tiles))

            trainer.split_train_val_data(root_tiles, val_fraction = results.val_fraction)

        trainer.train(multi_gpu = multi_gpu)

    main()

