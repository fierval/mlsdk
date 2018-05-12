from __future__ import print_function, division

import numpy as np
import pandas as pd
import json
import pandas as pd

from iuml.tools.image_utils import camera_from_file_name


def read_zones_from_json(zones_json):
    '''
    Read the JSON output of zone annotations and convert it to a 
    dictionary of Cam -> [zone-points]
    '''
    with open(zones_json, 'r') as f:
        image_descriptors = json.loads(f.read())
       
    # filter out by mark_count
    image_descriptors = \
        {camera_from_file_name(k): v for k,v in image_descriptors.items() if v['mark_count'] > 0}
    
    # remove fields that aren't needed
    for k, v in image_descriptors.items():
        del v['image_path']
        del v['mark_count']
        del v['quality']

    # drop the "marks" key and convert coordinates to simple tuples
    image_descriptors = {k: v['marks'] for k, v in image_descriptors.items()}
    return image_descriptors

def image_descriptors_to_dataframe(image_descriptors):
    '''
    Convert the output of read_zones_from_json to pandas dataframe
    '''
    dfs = []
    for k, d in image_descriptors.items():
        df = pd.DataFrame(d)
        df['Cam'] = k
        df['Zone'] = np.concatenate([[i] * 4 for i in range(3)])
        dfs += [df]
    df = pd.concat(dfs)
    df.drop(['class'], axis=1, inplace=True)

    # unsorted multi-index gives performance warnings
    polygons = df.set_index(['Cam', 'Zone']).sort_index()
    return polygons

if __name__ == '__main__':
    zones_json = r'C:\Users\boris\Dropbox (Personal)\iunu\data\smith\thoth\cams\cam_positions.json'   
    
    image_descriptors = read_zones_from_json(zones_json)
    camera_zones_frame = image_descriptors_to_dataframe(image_descriptors)

    