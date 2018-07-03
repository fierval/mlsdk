'''
Generators used to feed data to "predict"

These include:
- in-memory generator - for images already loaded into memory
- dir generator - for a directory of images
- url-generator - for images that exist as a list of URLs
'''

from __future__ import print_function, division, absolute_import

import numpy as np
import os
import abc
import cv2

from ..tools.image_utils import get_image_files
from ..tools.net import download_images

import keras.preprocessing.image as prep

def get_generator(batch_size, image_shape, preprocessing_function = None, data = None, dir = None, urls=None, capacity=50):
    '''
    Return generator requested
    '''
    if data is not None:
        return InMemoryImageGenerator(data, batch_size, image_shape, preprocessing_function, capacity)
    elif dir is not None:
        return DirImageGenerator(dir, batch_size, image_shape, preprocessing_function, capacity)
    elif urls is not None:
        return UrlImageGenerator(urls, batch_size, image_shape, preprocessing_function, capacity)
    else:
        raise ValueError("specify at least one generator")


class IumlPredictionGenerator(prep.Iterator):
    '''
    Abstract class for internal generator for predictions
    '''

    __metaclass__ = abc.ABCMeta
    def __init__(self, num_images, batch_size, image_shape, preprocessing_function = None, capacity = 50):
        '''
        Parameters:
            num_images - number of images
            batch_size - batch size
            image_shape - WxH image shape
            capacity - how many can we pre-load into memory
        '''
        if capacity < batch_size:
            raise ValueError("capacity should be greater than batch size")

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.capacity = min(capacity, num_images)
        self.need_reshape = None
        self.images = None
        self.preprocessing_function = preprocessing_function

        # tracks where we are in the internal batch
        self.internal_idx = self.capacity

        # False is for the shuffle parameter
        super(IumlPredictionGenerator, self).__init__(num_images, batch_size, False, None)

    def stack_images(self, image_list):
        if isinstance(image_list, np.ndarray):
            return image_list
        else:
            if self.need_reshape:
                image_list = [cv2.resize(im, self.image_shape) for im in image_list]
            return np.vstack(np.expand_dims(image_list, axis=0))

    def resize_if_needed(self, images):
        ims = images

        if self.need_reshape:
            ims = [cv2.resize(im, self.image_shape) for im in images]

        return self.stack_images(ims)

    @abc.abstractmethod
    def retrieve(self, index_array):
        '''
        Gets more images based on the index array
        (We never shuffle! This is a prediction generator)
        '''
        pass

    def _get_batches_of_transformed_samples(self, index_array):
        '''
        Override of keras Iterator.
        '''

        # index_array is always sequential (or looping) since we never shuffle!
        images = self.retrieve_next_internal_batch(index_array)
        batch = images[self.internal_idx : self.internal_idx + self.batch_size]

        self.internal_idx += self.batch_size

        return batch

    def set_need_reshape(self, image):
        '''
        Should we apply reshaping?
        '''
        self.need_reshape = True if image.shape[:2][::-1] != self.image_shape else False

    def _get_downloadable_range(self, index_array):

        leftover = self.capacity - self.internal_idx

        # need to get some more
        if leftover  < self.batch_size and len(index_array) > leftover:
            total_download = (self.capacity - leftover) % (self.num_images + 1)

            # if we are close to the end - just get remaining batches
            if index_array[leftover] + total_download > self.num_images:

                n_batches = (self.num_images - index_array[leftover] + self.batch_size - 1 + leftover) // self.batch_size
                total_download = n_batches * self.batch_size

                # this is the last time we are downloading.
                # shrink capacity for further calculations
                self.capacity = leftover + total_download

            idxs = np.mod(np.arange(index_array[leftover], index_array[leftover] + total_download), self.num_images)

            return idxs, leftover

        return None

    def retrieve_next_internal_batch(self, index_array):
        '''
        Given where we are internally, retrieve the next internal batch (up to capacity) if needed
        Parameters:
            index_array - next to retrieve
            retrieve_func - function that actually does the retrieval
        '''

        retrieve_range = self._get_downloadable_range(index_array)

        # for testing
        self.iter_idxs = index_array

        # retrieve
        if retrieve_range is not None:
            idxs, leftover = retrieve_range

            ims = self.retrieve(idxs)
            ims = self.resize_if_needed(ims)

            if self.preprocessing_function is not None:
                ims = self.stack_images([self.preprocessing_function(im) for im in ims])

            self.images = np.concatenate([self.images[-leftover:], ims]) if leftover > 0 else ims

            self.internal_idx = 0

        return self.images

    def next(self):
        '''
        Advancing iterator
        '''
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

class InMemoryImageGenerator(IumlPredictionGenerator):
    '''
    Feed images already in the memory to the model
    '''

    def __init__(self, images, batch_size, image_shape, preprocessing_function = None, capacity=None):
        '''
        Parameters:
            images - array-like of images
            batch_size - batch size
            image_shape - WxHxC image shape
            capacity - unused
        '''

        self.num_images = len(images)

        super(InMemoryImageGenerator, self).__init__(self.num_images, batch_size, image_shape, preprocessing_function, capacity=self.num_images)

        # verify if we need reshaping - can do it right away since everything is already loaded
        self.set_need_reshape(images[0])

        self.images = self.stack_images(images)

    def retrieve(self, index_array):
        return self.images[index_array]

class DirImageGenerator(IumlPredictionGenerator):
    '''
    Feed images already in the memory to the model
    '''

    def __init__(self, dir, batch_size, image_shape, preprocessing_function = None, capacity=50):
        '''
        Parameters:
            dir - path to images
            batch_size - batch size
            image_shape - WxH image shape
            capacity - should exeed or be equal to batch size
        '''

        if not os.path.exists(dir):
            raise FileExistsError("directory does not exist:{}".format(dir))

        # without np.array we can't address to it through
        #  self.image_files[index_array]
        self.image_files = np.array(get_image_files(dir))

        if len(self.image_files) == 0:
            raise FileNotFoundError("no files found in directory: {}".format(dir))

        self.num_images = len(self.image_files)

        super(DirImageGenerator, self).__init__(self.num_images, batch_size, image_shape, preprocessing_function, capacity)

        # verify if we need reshaping - can do it right away by just reading one file
        self.set_need_reshape(cv2.imread(self.image_files[0]))


    def retrieve(self, index_array):
        images = [cv2.imread(f) for f in self.image_files[index_array]]
        return images

class UrlImageGenerator(IumlPredictionGenerator):
    '''
    Feed images already in the memory to the model
    '''

    def __init__(self, urls, batch_size, image_shape, preprocessing_function, capacity=50):
        '''
        Parameters:
            urls - list of image urls
            batch_size - batch size
            image_shape - WxH image shape
            capacity - should exeed or be equal to batch size
        '''

        self.image_urls = np.array(urls)

        if len(self.image_urls) == 0:
            raise ValueError("empty image list")

        # images are reshaped when retrieved
        self.need_reshape = False
        self.num_images = len(self.image_urls)

        super(UrlImageGenerator, self).__init__(self.num_images, batch_size, image_shape, preprocessing_function, capacity)

    def retrieve(self, index_array):
        images = download_images(self.image_urls[index_array], img_shape = self.image_shape)
        return images

if __name__ == '__main__':
    from iuml.tools.train_utils import create_trainer
    import keras.backend as K
    import csv

    class TestGenerator(IumlPredictionGenerator):
        def __init__(self, num_images, batch_size, image_shape, capacity=100, need_reshape = True):
            '''
            Parameters:
                urls - list of image urls
                batch_size - batch size
                image_shape - WxH image shape
                capacity - should exeed or be equal to batch size
            '''

            # images are reshaped when retrieved
            self.need_reshape = need_reshape
            self.num_images = num_images
            shape = image_shape[::-1] + (3,)

            # generate random images
            self.test_ims = self.stack_images([np.array([[i]]) for i in range(num_images)])

            super(TestGenerator, self).__init__(self.num_images, batch_size, image_shape, capacity=capacity)

        def retrieve(self, index_array):
            images = self.test_ims[index_array]
            return images

    def run_case(number, num_images, batch_size, capacity):
        image_shape = (1, 1)

        print("Case: {}. Length: {}, batch: {}, capacity: {}".format(number, num_images, batch_size, capacity))

        gen = TestGenerator(num_images, batch_size, image_shape, capacity=capacity, need_reshape=False)
        n_batches = (num_images + batch_size - 1) // batch_size
        for i, batch in enumerate(gen):
            if i * batch_size >= num_images:
                break
            print("Batch: {}, start: {}, end: {}, internal idx: {}".format(i, gen.iter_idxs[0], gen.iter_idxs[-1], gen.internal_idx))
            print("Images: {}".format(batch.reshape(-1)))

        print("================ End Case =================")
        print("\n\n")

    def run_case_predict_dir():

        print("Predict from directory")
        model_file = r'C:\Users\boris\Dropbox (Personal)\iunu\data\n3xtgen\cnn\model_unet_2018_04_19.hd5'
        test_dir = r'C:\Users\boris\Dropbox (Personal)\iunu\data\n3xtgen\sampled_2018_04_27_18_41_56'
        batch_size = 1

        params = {'batch_size': batch_size, 'model_file': model_file}
        unet = create_trainer('Unet', '', **params)
        preds = unet.predict(dir = test_dir)
        K.clear_session()
        print("Predictions for {} images".format(preds.shape))
        print("================ End Case =================")
        print("\n\n")

    def run_case_predict_urls():

        print("Predict from URLs")
        model_file = r'C:\Users\boris\Dropbox (Personal)\iunu\data\n3xtgen\cnn\model_unet_2018_04_19.hd5'
        urls_file = r'C:\Users\boris\Dropbox (Personal)\iunu\data\n3xtgen\urls.csv'
        with open(urls_file, 'r') as uf:
            urls = np.concatenate(list(csv.reader(uf)))

        batch_size = 1

        params = {'batch_size': batch_size, 'model_file': model_file}
        unet = create_trainer('Unet', '', **params)
        preds = unet.predict(urls = urls)
        K.clear_session()
        print("Predictions for {} images".format(preds.shape))
        print("================ End Case =================")
        print("\n\n")

    def run_case_predict_data(num_images):

        print("Predict from data")
        data = [np.array([i % 255] * 300, dtype=np.uint8).reshape(10, 10, 3) for i in range(num_images)]
        batch_size = 32
        weights_file = r'C:\Users\boris\Dropbox (Personal)\iunu\data\buds\collection\smith\cnn\model_inception_v3.hd5'
        params = {'batch_size': batch_size, 'weights_file': weights_file}
        unet = create_trainer('InceptionV3', '', **params)
        preds = unet.predict(data=data, image_shape=(299, 299))
        K.clear_session()
        print("Predictions for {} images".format(preds.shape))
        print("================ End Case =================")
        print("\n\n")

    def main():
        # batch size same as capacity
        run_case(1, 24, 12, 12)

        ## batch size same as length
        run_case(2, 12, 12, 15)

        ## random batch size and length
        run_case(3, 81, 17, 38)

        ## length less than batch size
        run_case(4, 11, 20, 25)

        run_case_predict_data(59)

        run_case_predict_dir()

        run_case_predict_urls()

    main()

