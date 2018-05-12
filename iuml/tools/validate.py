from __future__ import print_function, division

''' Validation functions '''

import os

def raise_if_none(o, msg):
    if o is None or (type(o) == type("") and o == ''):
        raise ValueError(msg)

def is_str_none_or_empty(str):
    return str is None or str == ''

def raise_if_not_exists(dir):
    '''
    Raise exception if directory does not exist
    '''
    raise_if_none(dir, "Directory not specified")
    if not os.path.exists(dir):
        raise IOError("Directory does not exist: {}".format(dir))

def create_if_not_exists(dir):
    '''
    Create directory if it does not exist
    '''
    raise_if_none(dir, "Directory does not exist")
    if not os.path.exists(dir):
        os.makedirs(dir)

