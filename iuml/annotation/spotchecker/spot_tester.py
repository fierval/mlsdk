from __future__ import division, print_function
import os
import cv2
import numpy as np
from iuml.training import *

class SpotChecker(object):
    '''
    Allows to "spot-check" individual predictions given an image
    '''
    def __init__(self, image_file, model_file_path, net = 'VGG16', tile = 128, scale_factor = 2):
        '''
        Parameters:
            image -- the image to spot-check (actual loaded image)
            model -- path to the model file
            net -- network type
            tile -- size of the tile surrounding clicked point
            scale_factor -- inverse display factor
        '''

        self.DISPLAY_FACTOR = scale_factor
        self.INVERSE_DISPLAY_FACTOR = 1.0 / self.DISPLAY_FACTOR
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

        if image_file is None:
            raise ValueError("Image not present")

        if not os.path.exists(model_file_path):
            raise ValueError("Model file not found: {}".format(model_file_path))

        if scale_factor <= 0:
            raise ValueError("Scale factor should be positive")
        
        if tile <= 0:
            raise ValueError("Tile length should be positive")

        self.image = cv2.imread(image_file)
        self.radius = tile // 2
        
        params = dict(batch_size=1, weights_file = model_file_path, epochs=1, class_mode='binary')
        root_train = os.path.split(image_file)[0]
        self.trainer = create_trainer(net, root_train, **params)

        self.trainer.configure_model(multi_gpu = False)

    def preprocess_input(self, im):
        imcp = im.copy()
        imcp = cv2.resize(imcp, (224, 224)).astype(float)

        return preprocess_input(np.expand_dims(imcp, axis=0))

    def get_prediction_around(self, pt):
        '''
        Get prediction around the point
        '''

        col, row = pt

        cols = slice(col - self.radius, col + self.radius)
        rows = slice(row - self.radius, row + self.radius)
        
        im = self.image[rows, cols].copy()
        pred = self.trainer.predict(data=[im])
        return pred.ravel()[0]
        
    def click_event(self):
        def click_point(event, x_display, y_display, _flags, param):
            '''Click event. Get the region & predict'''
        
            if event == cv2.EVENT_LBUTTONDOWN:
                x_actual = int(self.DISPLAY_FACTOR * x_display)
                y_actual = int(self.DISPLAY_FACTOR * y_display)

                # for display only
                radius = int(self.radius * self.INVERSE_DISPLAY_FACTOR)
                lt = (x_display - radius, y_display - radius)
                rb = (x_display + radius, y_display + radius)

                prediction = "{:.3f}".format(self.get_prediction_around((x_actual, y_actual)))

                image = param['image']

                cv2.rectangle(image, lt, rb, (0, 255, 0), 3)
                cv2.putText(image, prediction, (x_display + radius, y_display), self.FONT, 1, (255, 255, 255), 2, cv2.LINE_AA)



        return click_point

    def check(self):
        display_image = cv2.resize(self.image, None, fx= self.INVERSE_DISPLAY_FACTOR, fy= self.INVERSE_DISPLAY_FACTOR)

        cv2.namedWindow('Checker')
        cv2.setMouseCallback('Checker', self.click_event(), param = {'image' : display_image})

        while True:
            cv2.imshow('Checker', display_image)
            key = cv2.waitKey(1) & 0xFF
    
            if key == 27: # ESC
                print('ESC pressed! quitting without saving')
                break
    
        cv2.destroyAllWindows()
