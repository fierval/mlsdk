from __future__ import print_function, division

from iuml.tools.image_utils import *
from iuml.tools.train_utils import create_trainer
from embedding import Embedder
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import glob
import warnings

class EmbedBinaryPredictions(Embedder):
    '''
    Create embedding based on prediction space
    '''

    def __init__(self, source, dest, model_file, model_type='VGG16', class_map = {1: 'positive', 0: 'negative'}, multi_gpu=False):
        '''
        Parameters:
            source - source images
            dest - destination embedding files
            model_file - (.hd5) model
            model_type - type of network (default: VGG16)
            class_map - map classes to meaningful names
            multi_gpu - should use multiple GPUs to compute predictions
            
        '''

        if not os.path.exists(model_file):
            raise ValueError("Model weights file not found")
        params = dict(model_file = model_file, batch_size = 32, class_mode='binary')
        # we don't really care what the model_type is since everything is loaded from model_file
        self.trainer = create_trainer(model_type, '', **params)

        # TODO: for binary models only
        self.positive = class_map[1]
        self.class_map = class_map
        self.model_file = model_file

        return super().__init__(source, dest)

    def expand_stack(self, arr):
        '''
        Expand the dims of each element of arr and stack vertically
        '''    
        lst = [np.expand_dims(a, axis=0) for a in arr]
        return np.vstack(lst)

    def create_embedding(self):
        
        files =  glob.glob(os.path.join(self.source, "**/*.jpg"))
        ims = [cv2.imread(f) for f in files]

        try:
            predictions = self.trainer.predict(data=ims)
        except:
            warnings.warn("Using weights file instead of a model file. This is deprecated", DeprecationWarning)

            self.trainer.model_file = None
            self.trainer.weights_file = self.model_file
            predictions = self.trainer.predict(data=ims)
        
        filenames = [os.path.split(f)[1] for f in files]
        classes = [os.path.split(os.path.split(f)[0])[1] for f in files]

        # if we are in the binary class mode
        # expand predictions vector into two dimensions
        print("Computing predictions...")
        if predictions[0].shape == (1,):
            p_list = [np.expand_dims(np.array([p[0], 1 - p[0]]), axis=0) for p in predictions]

        with tf.Session() as sess:
            _, images, file_paths = self.get_images_for_embedding(sess)
            images = [im for im in images]

            file_names = [os.path.split(fn)[1].decode() for fn in file_paths]
            class_names = [os.path.split(os.path.split(fn)[0])[1].decode() for fn in file_paths]

            # map images files to predictions files
            df_preds = pd.DataFrame({"pred": p_list, "files": filenames, 'class': classes})
            df_images = pd.DataFrame({"images": images, "files": file_names, 'class': class_names})

            df = pd.merge(df_preds, df_images, on="files")

            predictions = df['pred'].values
            images = self.expand_stack(df['images'].values)

            classes = df['class_x'].values
            file_names = df['files'].values

            
            self.create_embedding_actual(sess, predictions, images, file_names, classes)                        
            
    
    def create_embedding_actual(self, sess, predictions, images, file_names, classes):
        meta_name = 'labels.tsv'
        meta_path = os.path.join(self.dest, meta_name)
        print("Writing metadata...")
        self.write_metadata(meta_path, file_names, predictions, classes, thresh = 0.5, positive = self.positive)

        super().create_embedding_actual(sess, np.vstack(predictions), images, file_names, meta_path = meta_path, write_metadata = False)
   

    def write_metadata(self, meta_path, file_names, preds, classes, thresh = 0.5, positive = 'positive'):
        '''
        | Class | File | Pos | Neg | 
        '''
        
        init = np.repeat("--", len(preds))

        pos = init.copy()
        neg = init.copy()

        for i, (classs, scores) in enumerate(zip(classes, preds)):
            score = scores[0, 0]
 
            if score > thresh:
                if classs == positive:
                    pos[i] = "TP"
                else:
                    pos[i] = "FP"      
            else:
                if classs != positive:
                    neg[i] = "TN"
                else:
                    neg[i] = "FN"

        df = pd.DataFrame({'Class': classes, 'File': file_names, 'Pos': pos, 'Neg': neg})
        df.to_csv(meta_path, sep='\t', index=False)

