from __future__ import division, print_function

if __name__ == '__main__':

    import argparse
    from clusterView import ClusterViewer

    def parse_args():
        arg_parser = argparse.ArgumentParser()

        arg_parser.add_argument('--images', action='store', help='images directory')
        arg_parser.add_argument('--masks', action='store', help='masks directory')
        results = arg_parser.parse_args()
        return results, arg_parser


    def main():
        results, arg_parser = parse_args()

        images_dir = results.images
        masks_dir = results.masks

        viewer = ClusterViewer(images_dir, masks_dir)

        viewer.walk_images()

    main()
