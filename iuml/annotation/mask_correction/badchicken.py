
import os
try:
    import tkinter as tk
except:
    import Tkinter as tk
import pygubu
import cv2
import numpy as np
import pprint
import json
import logging
import Image, ImageTk
thisdir = os.path.abspath(os.path.dirname(__file__))

import logging

class BadChicken:

    def __init__(self):
        self.builder = builder = pygubu.Builder()

        builder.add_from_file(os.path.join(thisdir, 'badchicken.ui'))
        self.mainwindow = builder.get_object('Toplevel_1')
        self.builder.connect_callbacks(self)
        self.active_widgets = ['ImgCanvas', 
                               'PrevButton',
                               'NextButton',
                               'SaveButton',
                               'FilenameLabel',
                               'FilePicker'
                               ]
        def dbgattr(name):
            print(name)
            setattr(self, name, self.builder.get_object(name))
        [dbgattr(name) for name in self.active_widgets]
        self.imagepath = None
        self.maskpath = None
        self.pwd = None
        self.file_index = None
        self.working_mask = None
        self.pilimage = None
        self.tkimage = None
        self.all_images = None
        self.all_masks = None
        self.all_files = None
        self.display_scale_factor = None
        self.canvasimg = None
    def on_canvas_click(self,event):
        print "clicked at", event.x, event.y

        self.fill_mask(event.x/self.display_scale_factor, event.y/self.display_scale_factor )

        self.refresh_img()
        print event.x/self.display_scale_factor, event.y/self.display_scale_factor 
    def fill_mask(self, x, y):
        h, w = self.working_mask.shape[:2]
        i_hate_opencv = np.zeros((h+2, w+2), np.uint8)
        i, self.working_mask, hate, opencv = cv2.floodFill(self.working_mask, i_hate_opencv, (x,y), 0)
    def refresh_img(self):

        cv2img = self.combineimages(self.imagepath, self.maskpath, self.working_mask)
        print(cv2img.shape)
        self.ImgCanvas.delete('all')
        self.pilimage = Image.fromarray(cv2img)
        self.ImgCanvas.winfo_height()/2 
        self.display_scale_factor= factor = min(self.ImgCanvas.winfo_width()/self.pilimage.height, self.ImgCanvas.winfo_width()/self.pilimage.width)
        self.pilimage = self.pilimage.resize((self.pilimage.width*factor, self.pilimage.height*factor))
        self.tkimg = ImageTk.PhotoImage(image=self.pilimage)
        self.canvasimg= self.ImgCanvas.create_image(0, 0 , anchor=tk.NW, image=self.tkimg )


    def load(self, path=None):
        
        if path is None:
            thispath = self.FilePicker.cget('path')
        else:
            thispath = path

        print thispath
        if os.path.dirname(thispath) != self.pwd:

            self.pwd = os.path.dirname(thispath)

            self.all_files = listing = [ os.path.join(self.pwd,f) for f in os.listdir(os.path.dirname(thispath))]
            self.all_images = [os.path.join(self.pwd,f) for f in listing if not 'mask' in f]
            self.all_masks = [os.path.join(self.pwd,f) for f in listing if 'mask' in f]
            
        else:

            listing = self.all_files

        title = os.path.basename(thispath)
        self.file_index = self.all_images.index(thispath)
        print(self.file_index)

        try:
            
            masktitle = [f for f in self.all_masks if ((title[0:-4] in f) and (f != title))][0]
            self.imagepath = thispath
            self.maskpath = os.path.join(os.path.dirname(thispath),masktitle)
            self.refresh_img()

        except IndexError as e:
            
            print e
            print 'No corresponding mask file found. This is your fault.'

            return

    def next_image(self):
        self.file_index += 1
        self.working_mask = None
        self.load(self.all_images[self.file_index])
    def prev_image(self):
        self.file_index -= 1
        self.working_mask = None
        self.load(self.all_images[self.file_index])
    def save_image(self):
        cv2.imwrite(self.maskpath, self.working_mask)
    def combineimages(self, imagepath, maskpath, mask = None):

        if mask is None:
            mask = cv2.imread(maskpath, 0)
            _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)
        self.working_mask = mask
        
        img = cv2.cvtColor(cv2.imread(imagepath, 1), cv2.COLOR_BGR2RGB)
        


        img[mask>=240]/=2
        return img



    def quit(self,event=None):
        self.mainwindow.quit()
    
    def run(self):
        self.mainwindow.mainloop()


if __name__ == '__main__':
    app = BadChicken()
    app.run()


