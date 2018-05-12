from __future__ import print_function

import numpy as np
import cv2
import matplotlib.pylab as plt
from randomcolor import RandomColor
import re

def get_random_rgb(hue=None, luminosity=None, n=5):
    '''
    Get random colors as a list of dictionaries: 'r', 'g', 'b'
    '''
    rndclr = RandomColor()
    random_colors_strings = rndclr.generate(hue=hue, luminosity=luminosity, count=n, format_='rgb')

    # RandomColor.generate() returns a string: 'rgb(r, g, b)' for each entry. 
    # Unpack & transform to dict
    pattern = r'rgb\((?P<r>[0-9]+),\s(?P<g>[0-9]+),\s(?P<b>[0-9]+)\)'

    colors = np.concatenate([re.findall(pattern, c) for c in random_colors_strings])
    colors = [{'r':int(r), 'g':int(g), 'b':int(b)} for r, g, b in colors]
    return colors

def show_image(bgr_img, ax=None, title=None):
    '''
    Show image.
    Parameters:
        bgr_img - image as a numpy array loaded with OpenCV (8UC1 or 8UC3)
        ax - if specified, axis used for display
        title - if specified - adds a title
    '''
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB) if bgr_img.ndim > 2 else bgr_img
    if ax is not None:
        ax.axis('off')
        if title is not None:
            ax.set_title(title)
        ax.imshow(img)
    else:
        plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.imshow(img)

def mask_gray2bgr3(mask):
    '''
    Converts mask with 3 classes to a BGR image
    '''
    blue = np.zeros_like(mask)
    blue [mask == 0] = 255
    green = np.zeros_like(mask)
    green [mask == 2] = 255
    red = np.zeros_like(mask)
    red [mask == 1] = 255

    return np.stack((blue, green, red), axis=2).astype(np.uint8)

def mask_gray2bgr(mask, invert = False):
    '''
    Converts mask with 2 classes to a BGR image
    '''
    blue = np.zeros_like(mask)
    red = np.zeros_like(mask)
    green = np.zeros_like(mask)

    green[mask > 0] = 255
    red[mask == 0] = 255

    mask = (blue, green, red) if not invert else (blue, red, green)
    return np.stack(mask, axis=2).astype(np.uint8)

def mask_gray2bgr_random(mask):
    classes = np.unique(mask.ravel())
    colors = get_random_rgb(n=len(classes))
    green = np.zeros_like(mask)
    red = green.copy()
    blue = red.copy()

    for classs, color in zip(classes, colors):
        green[mask==classs] = color['g']
        red[mask == classs] = color['r']
        blue[mask == classs] = color['b']
    
    return np.stack((blue, green, red), axis=2).astype(np.uint8)        
    

def mask_convert(mask, invert = False):
    '''
    Depending on the number of classes, pick the right conversion routine
    '''
    classes = np.unique(mask.ravel()).size

    if classes <= 2:
        return mask_gray2bgr(mask, invert)
    elif classes == 3:
        return mask_gray2bgr3(mask)
    else:
        return mask_gray2bgr_random(mask)

def show_image_and_mask(image, mask, invert = False):
    '''
    image - RGB or grayscale image
    mask - RGB or grayscale mask
    invert - a mask of 0s and 1s only: whether to invert green and red (default - False)
    '''
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2,), (1, 0), colspan=2)

    if len(mask.shape) == 2:
        test_mask = mask_convert(mask, invert)
    else:
        test_mask = mask

    test_overlayed = cv2.addWeighted(image, 0.7, test_mask, 0.3, 0)
    show_image(image, ax=ax1)
    show_image(test_mask, ax=ax2)
    show_image(test_overlayed, ax=ax3)

def draw_grid_of_images(images, cols = 4, titles = None):
    '''
    Draws images on a grid of rows x cols
    (nice to look at in a Jupyter notebook)
    Parameters:
        images - array-like images to display
        cols - # of images / row. Default: 4
        titles - if specified - a function f(i)
                where i - is the order of an image, 
                returning a title
    '''
    length = len(images)
    rows = (length + cols - 1) // cols
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx >= length:
                break
            ax = plt.subplot2grid((rows, cols), (i, j))
            show_image(images[idx], ax=ax, title= None if titles is None else titles(idx))
