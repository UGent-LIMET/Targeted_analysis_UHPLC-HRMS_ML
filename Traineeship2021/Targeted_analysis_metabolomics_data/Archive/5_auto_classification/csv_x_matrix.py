
#!/usr/bin/env python 


from __future__ import with_statement 
from PIL import Image 
import os
import pandas as pd

PATH = '/media/sf_SF/Stage2021/Projects_ML_test/' #path to images

os.chdir(PATH)
df= pd.DataFrame()
with open(PATH + 'compare_names.csv', 'w+') as f: 
    for (dirpath, dirnames, filenames) in os.walk(PATH):
        for filename in filenames:
            if filename.endswith('.png'): 
                image = (dirpath + "/" + filename)
                im = Image.open(image)

                #load the pixel info 
                pix = im.load() 

                # get image name 
                image_name = filename.split("/")[-1] 
                image_name = image_name.split("_")
                image_name = image_name[1].split(".")[0] + "_" + image_name[3].split(".")[0]

                #get a tuple of the x and y dimensions of the image 
                width, height = im.size 
                f.write(image_name+',') 
                
                #read the details of each pixel and write them to the file 
                for x in range(width): 
                    for y in range(height): 
                        r = pix[x,y][0] 
                        g = pix[x,x][1] 
                        b = pix[x,x][2] 
                        f.write('{0},{1},{2},'.format(r/255,g/255,b/255)) 
                f.write('\n')
                         






