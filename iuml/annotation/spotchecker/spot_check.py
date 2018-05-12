from __future__ import print_function
import os
import argparse
from os import path
import cv2

def get_spot_checker():
    from spot_tester import SpotChecker
    return SpotChecker

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--image', action="store", default="", help='image to be checked')
    arg_parser.add_argument('--model', action='store', help='model weights file (.hd5)')
    arg_parser.add_argument('--scale-down', action='store', help='by how much should the image be scaled down', type=int, default=1)
    arg_parser.add_argument('--tile', action='store', help='tile side length', type=int, default=64)
    arg_parser.add_argument('--net', action='store', help='network (default: VGG16)', default='VGG16')

    results = arg_parser.parse_args()
    return results, arg_parser

def main():
    results, arg_parser = parse_args()

    image_file = results.image
    model_file = results.model
    tile = results.tile
    scale_down = results.scale_down
    net = results.net

    if image_file is None or model_file is None:
        arg_parser.print_usage()
        return

    if not path.exists(image_file):
        raise ValueError("Image does not exist")

    if not path.exists(model_file):
        raise ValueError("Model file does not exist")

    checker = get_spot_checker()(image_file, model_file, net = net, tile = tile, scale_factor = scale_down)
    checker.check()

if __name__ == '__main__':
    main()