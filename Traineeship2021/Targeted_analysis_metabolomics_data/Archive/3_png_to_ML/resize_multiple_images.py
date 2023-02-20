#!/usr/bin/python3

from PIL import Image
import os

def resize_multiple_images(src_path, dst_path):
    # Here src_path is the location where images are saved.
    for filename in os.listdir(src_path):
        if filename.endswith("_3.png"): 
            img=Image.open(src_path+filename)
            new_img = img.resize((64,64)) 
            new_img.save(dst_path+filename)
            