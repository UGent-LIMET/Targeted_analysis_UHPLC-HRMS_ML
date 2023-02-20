from PIL import Image
import os
import numpy as np
import re

def get_data(path):
    all_images_as_array=[]
    label=[]
    for filename in os.listdir(path):
        try:
            if re.match(r'FOUND',filename): #re.search ipv re.match? 
                label.append(1)
            else:
                label.append(0)
            img=Image.open(path + filename)
            np_array = np.asarray(img)
            l,b,c = np_array.shape    
            np_array = np_array.reshape(l*b*c,)   
            all_images_as_array.append(np_array)
        except:
            print(filename) #if error with 2dim, print
            continue

    return np.array(all_images_as_array), np.array(label)