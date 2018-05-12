from __future__ import print_function, division

if __name__ == '__main__':
    import argparse
    import os

    def create_embedding(source, destination, net = None, model_file = None, class_map = None, multi_gpu=False):
        '''
        Create embedding of the images
        '''

        from embedding import Embedder
        from prediction_embedding import EmbedBinaryPredictions


        if model_file is not None:
            embed = EmbedBinaryPredictions(source, destination, model_file, net, class_map, multi_gpu)
        else:
            embed = Embedder(source, destination)

        embed.create_embedding()

    def parse_args():
        arg_parser = argparse.ArgumentParser()

        arg_parser.add_argument('--source', action = 'store',
                    help='root directory for all the image data, split into subdirectories for classes')

        arg_parser.add_argument('--destination', action='store', help='where embedding will be stored')
        arg_parser.add_argument('--model', action='store', help='model file for visualizing predictions')
        arg_parser.add_argument('--net', action='store', help='network used to create the model (default: VGG16). InceptionV3 and Xception also supported', default='VGG16')
        arg_parser.add_argument('--class-map', action='store', help='maps integer values to class names', default='{1: "positive", 0: "negative"}')
        arg_parser.add_argument('--multi-gpu', action='store_true', help='use multiple GPUs if available')

        results = arg_parser.parse_args()
        return results, arg_parser

    def main():
        results, arg_parser = parse_args()

        if results.source is None or results.destination is None:
            arg_parser.print_usage()
            return

        if not os.path.exists(results.source):
            raise ValueError("Source path does not exist")

        destination = results.destination
        source = results.source
        model_file = results.model
        net = results.net
        class_map = eval(results.class_map)
        multi_gpu = results.multi_gpu

        if not os.path.exists(destination):
            os.makedirs(destination)

        if model_file is not None and not os.path.exists(model_file):
            raise ValueError('Model does not exist: {}'.format(model_file))

        if model_file is None and net is not None:
            print("Weights file not specified, computing image embedding")
            net = None

        create_embedding(source, destination, net, model_file, class_map, multi_gpu)

    main()