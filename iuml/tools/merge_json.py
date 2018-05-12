from __future__ import print_function
import os
from os import path
import json
import argparse

def arg_parse():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--folder', action="store",help='JSON files. This path will be used for all images')
    arg_parser.add_argument('--first', action='store', help='first JSON file')
    arg_parser.add_argument('--second', action='store', help='second JSON file')
    results = arg_parser.parse_args()
    return results, arg_parser

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def main():
    results, arg_parser = arg_parse()
    
    folder = results.folder
    first = results.first
    second = results.second

    if folder is None or first is None or second is None:
        arg_parser.print_usage()
        return

    source_file = path.join(folder, first)
    dest_file = path.join(folder, second)

    with open(source_file) as fp :
        source = json.load(fp)
    
    [new_im_folder, _] = path.split(list(source.items())[0][1]['image_path'])

    with open(dest_file) as fp:
        second = json.load(fp)

    source = merge_two_dicts(source, second)

    for key, val in source.items():
        _, im_file_name = path.split(val['image_path'])
        val['image_path'] = path.join(new_im_folder, im_file_name)

    with open(source_file, 'w') as outfile:
        outfile.write(json.dumps(source, indent=4, separators=(',', ': ')))

if __name__ == '__main__':
    main()