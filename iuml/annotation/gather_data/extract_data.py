# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 17:09:46 2017

@author: iunu
"""
from __future__ import print_function

if __name__ == '__main__':
    from annotator import Annotator
    from tile import tile_and_resize_images
    import os
    import argparse

    def parse_args():
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('--root-source', action="store", default="", help='root directory under which images sub-folders are stored')

        arg_parser.add_argument('--root-dest', action="store", help='root directory under which extracted tiles will be stored')
        arg_parser.add_argument('--tile', action='store', type=int, default=32, dest='length',
                                help='size of the square tile extracted in pixels of the original image (Default: 32)')
        arg_parser.add_argument('--scale-down', action='store', help='scale down factor applied before display', type=float, default=1)
        arg_parser.add_argument('--annotations-file', action='store', help='name of the annotations file. (Default: image_descriptors.json', default= 'image_descriptors.json')
        arg_parser.add_argument('--mask-scale', action='store', help='how much to scale the masks if "s" is pressed', default=0.25, type=float)

        results = arg_parser.parse_args()
        return results, arg_parser

    def main():
        results, arg_parser = parse_args()

        root_dest = results.root_dest
        root_source = results.root_source
        json_file = results.annotations_file
        mask_scale = results.mask_scale
        tile = results.length

        if root_source is None or root_source == "":
            print(arg_parser.print_usage())
            return

        if not os.path.exists(root_source):
            raise FileNotFoundError('Dataset root directory or any of its subdirectories do not exist')

        length = results.length
        scale_down = results.scale_down

        if scale_down < 1:
            raise ValueError("Scale down factor should be greater than 1")

        print('Starting positive extraction')

        annotator = Annotator(root_source, descriptor_file=json_file, radius=tile // 2, scale_factor = scale_down, mask_scale=mask_scale)

        if not annotator.run(): return

        if root_dest is not None:
            if not os.path.exists(root_dest):
                raise FileExistsError("Destination for tiles storage does not exist")

            annotator.generate_tiles(root_dest)

    main()
