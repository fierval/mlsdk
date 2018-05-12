from __future__ import print_function, division

import json

import pandas as pd
import numpy as np
import six

def points_from_json(zones_json):
    '''
    Read the JSON output of zone annotations and convert it to a 
    dictionary of File -> [(point, class)]
    '''
    with open(zones_json, 'r') as f:
        image_descriptors = json.loads(f.read())
       
    # filter out by mark_count
    image_descriptors = \
        {k: v for k,v in six.iteritems(image_descriptors)}
    
    # remove fields that aren't needed
    for k, v in six.iteritems(image_descriptors):
        del v['image_path']
        del v['mark_count']
        del v['quality']

    # drop the "marks" key and convert coordinates to simple tuples
    image_descriptors = {k: v['marks'] for k, v in image_descriptors.items()}
    return image_descriptors

def image_descriptors_to_dataframe(image_descriptors):
    '''
    Convert a dictionary of file -> [points] to a dataframe
    Parameters:
        image_descriptors - input dictionary
    Returns:
        Dataframe:
           File, x, y, class of points extracted from the dictionary
    '''
    dfs = []
    for k, d in six.iteritems(image_descriptors):
        df = pd.DataFrame(d)
        df['File'] = k
        dfs += [df]
    df = pd.concat(dfs)

    return df

