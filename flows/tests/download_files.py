
from __future__ import print_function, division, absolute_import

import numpy as np
from iuml.tools.auxiliary import *
from iuml.tools.image_utils import get_image_files
import time

def run_test(config, common, logger):

    title = config['title']
    facility = config['facility']

    collection = connect_prod_collection(common)

    logger.info("Downloading images from directory {}".format(config['dir']))
    start = time.time()
    get_images_from_dir(title, facility, collection, config['dir'], img_shape=eval(config['img_shape']), out_dir=config['out_dir'])
    logger.info("Download complete: {:.3f}".format(time.time() - start))