'''
python run_test.py config.json test_name

1. Create the test script in the same directory as run_test.py
2. Name the script: "my_test"
3. Add: `import my_test` to this file
4. Create "my_test.py" script to run the test.
   The test should be wrapped into `run_test(common, config, logger)` function (see examples)
5. Modify config.json by adding a section called "my_test" (see config.json example)
'''

from __future__ import print_function
import os
import argparse
from os import path
import cv2
import nextg3nbuds, download_files, select_images
import json
from datetime import datetime
from iuml.tools.auxiliary import *

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('config', action="store", default="", help='JSON configuration file')
    arg_parser.add_argument('name', action="store", default="", help='JSON configuration file')
    return arg_parser.parse_args(), arg_parser

def main():
    results, arg_parser = parse_args()
    
    if results.config is None or results.name is None:
        arg_parser.print_usage()
        return
    
    with open(results.config, 'r') as conf:
        all_config = json.load(conf)

    now = datetime.now()
    config = all_config[results.name]    
    common = all_config['common']

    if not os.path.exists(common['log_dir']):
        os.makedirs(common['log_dir'])

    log_file = os.path.join(common['log_dir'], results.name + "_{}.{}".format(now.strftime(common['current_time_suffix']), "log"))

    logger = create_logger(results.name, log_file= log_file, console=True)    
    eval(results.name).run_test(config, common, logger)

if __name__ == '__main__':
    main()