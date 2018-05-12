
from __future__ import division, print_function

if __name__ == '__main__':

    import argparse
    from recorder import Recorder

    def parse_args():
        arg_parser = argparse.ArgumentParser()

        arg_parser.add_argument('--image', action='store', help='image to process')
        arg_parser.add_argument('--mask', action='store', help='image mask')
        arg_parser.add_argument('--video', action='store', help='video file')
        arg_parser.add_argument('--fps', action='store', type=float, default=30, help='frames per second (default: 30)')
        arg_parser.add_argument('--scale', action='store', type=float, default=1., help='scale each side of the image by... (default: 1)')
        arg_parser.add_argument('--initial', action='store', type=int, default=8, help='initial buds for animation (default: 8)')
        arg_parser.add_argument('--q', action='store', type=float, default=2, help='progression coefficient (default: 2)')
        arg_parser.add_argument('--all-at-once', action='store_true')
        results = arg_parser.parse_args()
        return results, arg_parser


    def main():
        results, arg_parser = parse_args()

        image_file = results.image
        image_mask = results.mask
        video = results.video
        q = results.q
        initial = results.initial
        all_at_once = results.all_at_once

        fps = results.fps
        scale = results.scale

        if q <= 1:
            raise ValueError("q should be greather than 1")
        if initial <= 0:
            raise ValueError("initial should be positive")

        rec = Recorder(image_file, image_mask, video, fps, scale, initial, q)
        rec.record(all_at_once)

    main()