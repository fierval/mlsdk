import multiprocessing
from multiprocessing import pool
import requests
import os
import cv2
import numpy as np

def download_images(image_urls, img_shape = None, out_dir = None):
    '''
    Download a list of img urls
    Parameters:
        img_urls - array-like of image urls
        img_shape - shape to resize images to
    Returns:
        Map of output file -> input URL
    '''

    def download_one(url_order):
        '''
        Download a single file
        '''
        url, i = url_order
        r = requests.get(url, timeout=60)
        buf = bytearray(r.content)
        try:
            encimg = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            im = cv2.resize(encimg, img_shape, interpolation=cv2.INTER_CUBIC) if img_shape is not None else encimg

            if i % 10 == 0:
                print("Downloaded {}\n".format(i))
        
            if out_dir is not None and os.path.exists(out_dir):
                file_name = os.path.join(out_dir, url[url.rfind('/') + 1:])
                cv2.imwrite(file_name, im)

            return im
        except:
            return np.zeros(img_shape[::-1] + (3,))
            
    downloader = pool.ThreadPool(multiprocessing.cpu_count())
    # we only need the "i" parameter for outputting feedback: 'image i downloaded'
    return downloader.map(download_one, [(url, i) for i, url in enumerate(image_urls)])

