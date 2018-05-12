from __future__ import print_function, division

import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import math
from fnmatch import fnmatch
from tensorflow.contrib.tensorboard.plugins import projector
import pandas as pd
from iuml.tools.image_utils import get_image_files, image_exts

class Embedder(object):
    '''
    Wraps Tensorflow Embedding
    '''

    def __init__(self, source, dest):
        '''
        Parameters:
            source - source directory where image sub-directories exist with one class of images per subdirectory
            dest - destination embedding files
        '''
        self.source = source
        self.dest = dest
        self._classes = {}

        if not (os.path.exists(source) and os.path.exists(dest)):
            raise ValueError("Source and destination paths must exist")
    @property
    def classes(self):
        if len(self._classes) == 0:
            self.get_classes_from_subdirectories()
        return self._classes

    def create_embedding_actual(self, sess, vecs, images, file_paths, meta_path = None, write_metadata = False):
        embedded = tf.Variable(vecs, name='embeddings')
        saver = tf.train.Saver([embedded])

        # create labels
        if meta_path is None:
            meta_name = 'labels.tsv'
            meta_path = os.path.join(self.dest, meta_name)

        # metadata should be written independently to this fiile
        if write_metadata:
            print("Writing metadata...")
            self.write_metadata(meta_path, file_paths)
        
        # create sprite
        sprite_name = 'sprite.png'
        sprite_path = os.path.join(self.dest, sprite_name)
        print("Saving sprite image...")
        sprite = self.images_to_sprite(images)
        cv2.imwrite(sprite_path, cv2.cvtColor(sprite, cv2.COLOR_BGR2RGB))
        
        ### Embedding boilerplate ###
        # create actual embedding
        sess.run(embedded.initializer)
        checkpoint_path = os.path.join(self.dest, 'embeddings.ckpt')
        print("Saving embedding checkpoing: {}".format(checkpoint_path))
        saver.save(sess, checkpoint_path)
        
        # write configuration
        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter(self.dest)
        
        embedding = config.embeddings.add()
        embedding.tensor_name = embedded.name
        
        # meatadata & images
        embedding.metadata_path = meta_path

        embedding.sprite.image_path= sprite_path
        embedding.sprite.single_image_dim.extend(self.sample_image.shape[:2])
        
        projector.visualize_embeddings(summary_writer, config)
        tf.summary.merge_all()
        print("Done!")

    def create_embedding(self):
        with tf.Session() as sess:
            vecs, images, file_paths = self.get_images_for_embedding(sess)
            # Create the embedding
            self.create_embedding_actual(sess, vecs, images, file_paths, write_metadata = True)            

    def get_images_for_embedding(self, sess):
        '''
        Embed image vectors
        '''
        
        print("Start embedding...")
        files_pattern = os.path.join(self.source, '**/*.jpg')
        # get all the files from all subdirectories
        matched_files = tf.train.match_filenames_once(files_pattern)
        
        # total files
        # file_count = tf.size(matched_files)
        filename_queue = tf.train.string_input_producer(matched_files)
        num_files = 0
        num_files += len(glob.glob(os.path.join(self.source, '**/*.jpg')))

        image_reader = tf.WholeFileReader()
        self.sample_image = self.get_first_example()
        
        tensor_shape = self.sample_image.shape

        # create coordinator based on the file name queue
        fname, image_file = image_reader.read(filename_queue)
        image_tensor = tf.image.decode_jpeg(image_file)
        image_tensor.set_shape(tensor_shape)

        batch_size = num_files
        num_preprocess_threads = 1
        min_queue_examples = batch_size

        # tensor to read images in batches
        image_fname, image_batch = tf.train.batch(
            [fname, image_tensor],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            allow_smaller_final_batch=True)

        # initialize all variables
        _ = sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    
        # coordinator
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord=coord)
        print("Found {} files in {} classes".format(num_files, len(self.classes)))

        try:
            file_paths, images_for_sprite = sess.run([image_fname, image_batch])
        finally:
            coord.request_stop()
            coord.join(threads)
            
        print("Done computing embedding...")
            
        vecs = images_for_sprite.reshape(num_files, -1)
        return vecs, images_for_sprite, file_paths

    def get_first_example(self):
        '''
        Read the very first image we can lay our hands on.
        Tensorflow needs exact tensor dimensions before it can batch them
        '''

        # loop over image extensions to find image files
        im = None

        for ext in image_exts():
            image_files = None
            for root, dirs, files in os.walk(self.source):
                if len(files) == 0: continue
                image_files = filter(lambda f: fnmatch(f, ext), files)
                break

            if image_files is None:
                raise ValueError('No files found in the source directory')

            # we only need the first file
            for f in image_files:
                im = cv2.imread(os.path.join(root, f))
                break

            if not im is None: break

        if im is None:
            raise ValueError('Failed to read image')
           
        return im

    def get_classes_from_subdirectories(self):
        '''
        Image classes and their sizes
        '''
        for _, dirs, _ in os.walk(self.source):
            break

        for classs in dirs:
            classs_path = os.path.join(self.source, classs)
            self._classes[classs] = len(glob.glob1(classs_path, '*.jpg'))

    def write_metadata(self, meta_path, file_paths):
        '''
        Writes labels file in the format:
        | Class | File |
        Parameters:
            meta_path - path to the labels file
            file_names - full paths of file names in order they were processed
        '''

        # gets the subdirectory of the file (class name)
        # and the actual file name
        class_names = [os.path.split(os.path.split(fn)[0])[1].decode() for fn in file_paths]
        file_names = [os.path.split(fn)[1].decode() for fn in file_paths]

        df = pd.DataFrame({'Class' : class_names, 'File' : file_names})
        df.to_csv(meta_path, sep='\t', index=False)

    def images_to_sprite(self, data):
        """Creates the sprite image along with any necessary padding

        Args:
          data: NxHxW[x3] tensor containing the images.

        Returns:
          data: Properly shaped HxWx3 image with any necessary padding.
        """
        if len(data.shape) == 3:
            data = np.tile(data[...,np.newaxis], (1,1,1,3))
        data = data.astype(np.float32)
        min = np.min(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
        max = np.max(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
        # Inverting the colors seems to look better for MNIST
        #data = 1 - data

        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, 0),
                (0, 0)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant',
                constant_values=0)
        # Tile the individual thumbnails into an image.
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        data = (data * 255).astype(np.uint8)
        return data
