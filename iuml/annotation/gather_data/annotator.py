#!/usr/bin/env python
'''This script lets a user place marks on a collection of images.'''

from __future__ import division, print_function
import json
import os
import fnmatch
import cv2
import numpy as np
import copy
import math
import multiprocessing as mp
from functools import partial
from iuml.tools.image_utils import get_image_files, image_exts
from iuml.tools.validate import create_if_not_exists





def dist(x1, y1, x2, y2):
    d = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return d



# REVIEW: We are using dummy multiprocessing here
# but it doesn't appear to give us any perf improvement
def tile_single_image(img, radius, out_file_template, mark_and_idx):
    i, mark = mark_and_idx

    if 'x2' not in mark:
        # only generate tiles for circle marks

        dir = mark['class']
        if 'r' in mark:
            radius = mark['r']

        cx = mark['x']
        cy = mark['y']
    
        x1, x2 = int(cx - radius), int(cx + radius)
        y1, y2 = int(cy - radius), int(cy + radius)
        x1 = max(0, x1)
        y1 = max(0, y1)
        tile = int(2 * radius)

        cropped = img[ y1:y2, x1:x2 ]
        
        out_file = out_file_template.format(dir, i + 1)

        if not cv2.imwrite(out_file, cropped):
            dir = os.path.split(out_file)[0]
            os.makedirs(dir)
            cv2.imwrite(out_file, cropped)



class Annotator(object):
    ''' 
    Run the UI to annotate images and extract samples.
    A sample is characterized by center & radius
    '''
    
    def __init__(self, root_path, descriptor_file = 'image_descriptors.json'
                , masks_folder = 'masks', radius = 100, scale_factor = 1, mask_scale = 0.25, scale_output = False):
        '''
        Parameters:
            root_path - root folder for all images
            images_folder - subfolder of root_path where image files reside
            radius - radius around the selected center
            scale_factor - how much the image should be scaled down before display
            scale_output - should extracted tiles be scaled down by the same factor as the display
        '''

        if scale_factor <= 0:
            raise ValueError("Scale factor should be positive")

        self.root_path = root_path
        self.images_folder = self.root_path
        self.desc_file_path = os.path.join(self.root_path, descriptor_file)
        self.mask_folder_path = os.path.join(self.root_path, masks_folder)
        self.display_factor = scale_factor
        self.inverse_display_factor = 1.0 / self.display_factor
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.radius = radius
        self.mask_scale = mask_scale

        self.scale_output = scale_output

        self.annotationsSaved = False        
        self.prev_point = None

        self.annotation_class = '1'



        
    def click_event(self):
        def click_point(event, x, y, _flags, param):
            '''Click event. It adds the coordinates of the clicked point
            to the marks list of the appropriate image descriptor.'''
        
            pack = param['run_pack']
            x = int(self.display_factor * x)
            y = int(self.display_factor * y)
            

            if pack['resize_marks']:
                if event == cv2.EVENT_LBUTTONDOWN:

                    if pack['rectangles']:
                        pack['temp_display_mark'] = {'x': x, 'y': y, 'x2': x+1, 'y2': y+1}
                    else:
                        pack['temp_display_mark'] = {'x': x, 'y': y, 'r': 1}

                    pack['temp_display_mark']['class'] = self.annotation_class


                elif event == cv2.EVENT_LBUTTONUP:

                    image_name = pack['image_name']
                    image_descriptors = pack['image_descriptors']
                    image_desc = image_descriptors[image_name]
                    
                    image_desc['marks'].append(pack['temp_display_mark'])

                    marks = image_desc['marks']
                    tdm = pack['temp_display_mark']
                    image_desc['mark_count'] += 1
                    pack['temp_display_mark'] = None

                else:
                    # mouse moved, but button not let up

                    if pack['temp_display_mark']:
                        tdm = pack['temp_display_mark']

                        if pack['rectangles']:
                            tdm['x2'] = x
                            tdm['y2'] = y
                        else:
                            tdm['r'] = dist(tdm['x'], tdm['y'], x, y)

 
            else:

                new_drag = False
                if pack['drag_points']:
                    if self.prev_point is None:
                        new_drag = True
                        self.prev_point = (x, y)
                    else:
                        px = self.prev_point[0]
                        py = self.prev_point[1]
                        d = dist(px, py, x, y)

                        if d > 100:
                            self.prev_point = (x, y)
                            new_drag = True

                if event == cv2.EVENT_LBUTTONDOWN or new_drag:

                    image_name = pack['image_name']
                    image_descriptors = pack['image_descriptors']
                    image_desc = image_descriptors[image_name]
            
                    marks = image_desc['marks']
                    r = self.radius

                    if pack['rectangles']:
                        marks.append({'x': x-r, 'y': y-r, 'x2': x+r, 'y2':y+r, 'class':self.annotation_class})
                    else:
                        marks.append({'x': x, 'y': y, 'class':self.annotation_class})


                    image_desc['mark_count'] += 1


        return click_point
    



    def draw_mark(self, image, mark, color, scale_factor, thickness):
        x = int(mark['x'] * scale_factor)
        y = int(mark['y'] * scale_factor)

        if 'x2' in mark:
            # it's a rectangle mark
            x2 = int(mark['x2'] * scale_factor)
            y2 = int(mark['y2'] * scale_factor)
            cv2.rectangle(image, (x, y), (x2, y2), color, thickness)
        else:
            # it's a circle mark
            if 'r' in mark:
                r = int(mark['r'] * scale_factor)
            else:
                r = int(self.radius * scale_factor)
            cv2.circle(image, (x, y), r, color, thickness)


    
    def draw_marks(self, pack):
        '''Uses the marks list in the appropriate image descriptor to draw all marks on
        a fresh display image.'''
    
        image_name = pack['image_name']
        desc = pack['image_descriptors']
        if image_name in desc:
            marks = desc[image_name]['marks']
            image = pack['fresh_display_image'].copy()
            pack['display_image'] = image
    
            for mark in marks:
                annotation_class = mark['class']
                if annotation_class == '0':
                    color = (255, 255, 0)                
                if annotation_class == '1':
                    color = (255, 0, 0)                
                if annotation_class == '2':
                    color = (0, 255, 0)
                if annotation_class == '3':
                    color = (0, 0, 255)
                if annotation_class == '4':
                    color = (148, 0, 211)
                if annotation_class == '5':
                    color = (100, 149, 237)

                self.draw_mark(pack['display_image'], mark, color, self.inverse_display_factor, 2)

            if pack['temp_display_mark']:
                self.draw_mark(pack['display_image'], pack['temp_display_mark'], (0,0,255), self.inverse_display_factor, 2)



            if pack['display_text']:
                mark_count = desc[image_name]['mark_count']
                quality = desc[image_name]['quality']
                annotation_class = self.annotation_class
    
                text = str(mark_count) + ' marks'
                cv2.putText(image, text, (20, 50), self.font, 1, (255, 255, 255), 3, cv2.LINE_AA)
    
                quality_text = 'quality:' + quality
                cv2.putText(image, quality_text, (20, 80), self.font, 1, (255, 255, 255), 3, cv2.LINE_AA)

                class_text = 'current class: ' + annotation_class
                cv2.putText(image, class_text, (20, 110), self.font, 1, (255, 255, 255), 3, cv2.LINE_AA)
    
    
    def get_image_name(self, path):
        '''Parses the path of an image to obtain the image name. '''
    
        basename = os.path.basename(path)
        return basename
    
    
    def get_previous_file_path(self, pack):
        '''Returns the previous path from the collection of paths. '''
    
        i_current = pack['paths'].index(pack['image_path'])
        if i_current > 0:
            pack['image_path'] = pack['paths'][i_current - 1]
    
    
    def get_next_file_path(self, pack):
        '''Returns the next path from the collection of paths. '''
    
        paths = pack['paths']
        try:
            i_current = paths.index(pack['image_path'])
            i_max = len(paths) - 1
            if i_current < i_max:
                pack['image_path'] = paths[i_current + 1]
        except Exception:
            pack['image_path'] = paths[0]
    
    
    def get_previous_marked_file_path(self, pack):
        '''Returns the first previous path to an image which already has an
        image descriptor.'''
    
        i_current = pack['paths'].index(pack['image_path'])
        for i in range((i_current - 1), -1, -1):
            path = pack['paths'][i]
            name = self.get_image_name(path)
            if name in pack['image_descriptors']:
                pack['image_path'] = path
                break
    
    def generate_scaled_marks(self, pack, scale_factor):
        '''Scales mark position based on the scale factor provided '''
        
        scaled = copy.deepcopy(pack['image_descriptors'])
        for (_name, desc_pack) in scaled.items():
            for mark in desc_pack['marks']:
                mark['x'] = int(scale_factor * mark['x'])
                mark['y'] = int(scale_factor * mark['y'])
    
        path = self.desc_file_path.replace('.json', '_' + str(scale_factor) + '.json')    
        with open(path, 'w') as outfile:
            outfile.write(json.dumps(scaled, indent=4, separators=(',', ': ')))



    def generate_tiles(self, tiles_path, prefix=''):
        '''
        Creates tiles from json data & save them
        '''
        if not os.path.exists(tiles_path):
            os.makedirs(tiles_path)
            
        # ensure correct data
        if not self.annotationsSaved:
            self.save(self.run_pack['image_descriptors'])
        
        # should not forget to up-scale radius!
        radius = self.radius

        # now save each tile
        for j, (name, desc_pack) in enumerate(self.run_pack['image_descriptors'].items()):
            out_file_name_template = os.path.join(tiles_path, '{}', os.path.splitext(name)[0] + prefix + '-{:04d}.jpg')            
            img = cv2.imread(desc_pack['image_path'])

            print("Tiling image: {0:d} of {1:d}".format(j + 1, len(self.run_pack['image_descriptors'])))

            if img is None:
                raise ValueError('could not load image {}'.format(desc_pack['image_path']))

            # if we are rescaling - rescale the image and the coordinates
            if self.inverse_display_factor != 1. and self.scale_output:
                img = cv2.resize(img, None, fx=self.inverse_display_factor, fy=self.inverse_display_factor)

                marks = []
                i = 0
                for mark in desc_pack['marks']:
                    mark['x'] = mark['x'] * self.inverse_display_factor
                    mark['y'] = mark['y'] * self.inverse_display_factor
                    if 'r' in mark:
                        mark['r'] = mark['r'] * self.inverse_display_factor
                    if 'x2' in mark:
                        mark['x2'] = mark['x2'] * self.inverse_display_factor
                        mark['y2'] = mark['y2'] * self.inverse_display_factor

                marks.append( (i, mark) )
                i += 1

            else:
                marks = [(i, mark) for i, mark in enumerate(desc_pack['marks'])]
                radius = int(self.radius * self.display_factor)

            func = partial(tile_single_image, img, radius, out_file_name_template)
            pool = mp.Pool(mp.cpu_count())
            
            pool.map(func, marks)
            pool.close()
            pool.join()



    def generate_masks(self, pack):
        ''' generates a mask image for each image with an image descriptor. '''
    
        # Create the masks directory if one doesn't exist
        create_if_not_exists(self.mask_folder_path)

        for (_name, desc_pack) in pack['image_descriptors'].items():
            path = desc_pack['image_path']
    
            basename = self.get_image_name(path)
            mask_name = basename
            mask_path = os.path.join(self.mask_folder_path, mask_name)

            print('mask_path: ' + str(mask_path))
    
            i = cv2.imread(path)
            h, w, d = i.shape
    
            mask = np.zeros((h, w), np.uint8)
            for mark in desc_pack['marks']:
                self.draw_mark(mask, mark, 255, 1.0, -1)
    
            cv2.imwrite(mask_path, mask)

    
    
    def get_unmarked_file_path(self, pack):
        ''' returns the path to an image with no image descriptor. '''
    
        for path in pack['paths']:
            name = self.get_image_name(path)
            if name in pack['image_descriptors']:
                continue
            else:
                print ('found an unmarked image, path: ' + path)
                pack['image_path'] = path
                break
    
    
    def load_image(self, pack):
        ''' loads all necessary information into an image pack.  It is assumed that the
        image_path property has been set to a new image. '''
    
        path = pack['image_path']
        image = cv2.imread(path)
        image_name = self.get_image_name(path)
        image_index = pack['paths'].index(path)
        display_image = cv2.resize(image, (0, 0), fx=self.inverse_display_factor, fy=self.inverse_display_factor)
    
        if pack['display_text']:
            text = str(image_index) + ' - ' + image_name
            cv2.putText(display_image, text, (20, 20), self.font, 1, (255, 255, 255), 3, cv2.LINE_AA)
    
        pack['image'] = image
        pack['image_name'] = image_name
        pack['fresh_display_image'] = display_image
        pack['display_image'] = display_image.copy()
    
        image_name = pack['image_name']
        image_descriptors = pack['image_descriptors']
        if image_name in image_descriptors:
            image_desc = image_descriptors[image_name]
            image_desc['image_path'] = pack['image_path']
        else:
            image_desc = {'image_path': pack['image_path'],
                          'marks': [],
                          'mark_count': 0,
                          'quality': 'None'}
            image_descriptors[image_name] = image_desc
    
    def load_run_pack(self, pack):
        ''' loads the run_pack from a json file found at DESC_FILE_PATH.'''
    
        image_descriptors = {}
        if os.path.isfile(self.desc_file_path):
            try:
                with open(self.desc_file_path) as in_file:
                    image_descriptors = json.loads(in_file.read())
    
                for key, file_descriptor in image_descriptors.items():
                    if 'quality' not in file_descriptor:
                        file_descriptor['quality'] = 'None'
            except:
                pass
    
        image = None
        display_image = None
        image_name = ''
        image_path = None
        paths = []
        for root, _dirnames, fnames1 in os.walk(self.images_folder):
            for ext in image_exts():
                fnames = fnmatch.filter(fnames1, ext)
                for fname in fnames:
                    paths.append(os.path.join(root, fname))
    
        pack['image_descriptors'] = image_descriptors
        pack['display_image'] = display_image
        pack['image'] = image
        pack['image_name'] = image_name
        pack['paths'] = paths
        pack['image_path'] = image_path
        pack['display_text'] = True
        pack['drag_points'] = False
        pack['resize_marks'] = False
        pack['temp_display_mark'] = None
        pack['rectangles'] = False




    def export_categorical_classes(self, descriptors):

        export_desc = copy.deepcopy(descriptors)
        
        for key, images_bag in export_desc.items():
            img_path = os.path.join(self.images_folder, key)
            images_bag['image_path'] = img_path


            keep_marks = []
            for i, mark in enumerate( images_bag['marks'] ):

                if 'x2' in mark:
                    # it's a rectangular mark
                    keep_marks.append(mark)

                    if mark['x'] < mark['x2']:
                        mark['xmin'] = mark['x']
                        mark['xmax'] = mark['x2']
                    else:
                        mark['xmin'] = mark['x2']
                        mark['xmax'] = mark['x']

                    if mark['y'] < mark['y2']:
                        mark['ymin'] = mark['y']
                        mark['ymax'] = mark['y2']
                    else:
                        mark['ymin'] = mark['y2']
                        mark['ymax'] = mark['y']

                    del mark['x']
                    del mark['y']
                    del mark['x2']
                    del mark['y2']

                    mark_class = [0, 0, 0, 0, 0, 0]
                    mark_class[int(mark['class'])] = 1
                    mark['class'] = tuple(mark_class)

            images_bag['marks'] = keep_marks

        #TODO: parameterize this instead of hard-coding!
        with open(os.path.join(os.path.split(self.desc_file_path)[0], 'categorical_classes.json'), 'w') as outfile:
            outfile.write(json.dumps(export_desc, indent=4, separators=(',', ': ')))



    def save(self, descriptors):
        ''' Save image descriptors to the JSON object
        '''
        self.annotationsSaved = True
        # fix file paths to point to actual images (in case we moved .json file)
        for key, images_bag in descriptors.items():
            img_path = os.path.join(self.images_folder, key)
            images_bag['image_path'] = img_path

        print("Serializing JSON object into {}".format(self.desc_file_path))
        with open(self.desc_file_path, 'w') as outfile:
            outfile.write(json.dumps(descriptors, indent=4, separators=(',', ': ')))

            
    def run(self):
        ''' This is the main loop. '''

        self.run_pack = {}
        self.load_run_pack(self.run_pack)
        self.get_next_file_path(self.run_pack)
        if self.run_pack['image_path'] is None:
            self.run_pack['image_path'] = self.run_pack['paths'][0]
    
        self.load_image(self.run_pack)
    
        cv2.namedWindow('window')
        cv2.setMouseCallback('window', self.click_event(), param={'run_pack' : self.run_pack})


        display_image = self.run_pack['display_image']
        image_descriptors = self.run_pack['image_descriptors']
        save_on_exit = False
        
        while display_image is not None:
            display_image = self.run_pack['display_image']
            self.draw_marks(self.run_pack)
            cv2.imshow('window', display_image)
            key = cv2.waitKey(1) & 0xFF
    
            if key == ord('p'):
                print('p pressed, getting previous image')
                self.get_previous_file_path(self.run_pack)
                self.load_image(self.run_pack)
    
            elif key == ord('n'):
                print('n pressed, getting next image')
                print('serializing json: {}'.format(self.desc_file_path))
                self.save(self.run_pack['image_descriptors'])

                self.get_next_file_path(self.run_pack)
                self.load_image(self.run_pack)
    
            elif key == ord('c'):
                print('c pressed, toggling resizable circle mode')
                self.run_pack['resize_marks'] = not self.run_pack['resize_marks']

    
            elif 81 <= key <= 84:
                print('arr key pressed.  tweaking circle position.')
                marks = image_descriptors[self.run_pack['image_name']]['marks']
                last = marks[-1]

                if key % 2 == 0:
                    # even key means up/down

                    d = key - 83
                    last['y'] = last['y'] + d
                else:
                    d = key - 82
                    last['x'] = last['x'] + d

                image_descriptors[self.run_pack['image_name']]['marks'] = marks
                self.load_image(self.run_pack)

            elif key == ord('b'):
                print ('b pressed, getting previous marked image')
                self.get_previous_marked_file_path(self.run_pack)
                self.load_image(self.run_pack)
    
            elif key == ord('x'):
                print ('x pressed, removing last mark')
    
                marks = image_descriptors[self.run_pack['image_name']]['marks']
                image_descriptors[self.run_pack['image_name']]['marks'] = marks = marks[:-1]
                image_descriptors[self.run_pack['image_name']]['mark_count'] = len(marks)
                self.load_image(self.run_pack)
    
            elif key == ord('g'):
                print ('g pressed, generating masks')
                self.generate_masks(self.run_pack)
    
            elif key == ord('s'):
                print ('s pressed, generating scaled descriptors')
                self.generate_scaled_marks(self.run_pack, self.mask_scale)


            elif key == ord('r'):
                print ('r pressed, rectangle mode!')
                self.run_pack['rectangles'] = not self.run_pack['rectangles']

    
            elif key == ord('q'):
                print ('Q pressed!  quitting and saving')
                save_on_exit = True
                break
    
            elif key == 27: # ESC
                print('ESC pressed! quitting without saving')
                save_on_exit = False
                break
            
            elif key == ord('d'):
                self.run_pack['display_text'] = not self.run_pack['display_text']
                self.load_image(self.run_pack)
    
            elif key == ord('+'):
                image_descriptors[self.run_pack['image_name']]['quality'] = 'good'
    
            elif key == ord('-'):
                image_descriptors[self.run_pack['image_name']]['quality'] = 'bad'
    
            elif key == ord('='):
                image_descriptors[self.run_pack['image_name']]['quality'] = 'average'

            elif key == ord('o'):
                self.run_pack['drag_points'] = not self.run_pack['drag_points']

            elif key == ord('0'):
                self.annotation_class = '0'

            elif key == ord('1'):
                self.annotation_class = '1'

            elif key == ord('2'):
                self.annotation_class = '2'

            elif key == ord('3'):
                self.annotation_class = '3'

            elif key == ord('4'):
                self.annotation_class = '4'

            elif key == ord('5'):
                self.annotation_class = '5'

    
        cv2.destroyAllWindows()
        
        if save_on_exit:
            print('serializing json: {}'.format(self.desc_file_path))
            self.save(self.run_pack['image_descriptors'])
            self.export_categorical_classes(self.run_pack['image_descriptors'])

        return save_on_exit

    @property
    def annotation_file(self):
        return self.desc_file_path