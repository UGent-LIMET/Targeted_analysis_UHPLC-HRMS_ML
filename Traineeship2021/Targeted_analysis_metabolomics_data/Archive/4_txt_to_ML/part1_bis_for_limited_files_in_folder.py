
# Explanation about the script 

############# Packages #############
####################################

import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import shutil
import random
from get_functions import files, directories
from transform_functions import txt_to_df, df_to_png, resize_images, get_data
from rename_functions import rename_files

############ Adjustments ############
#####################################

# project files
PATH = "/media/sf_SF/Stage2021/Projects/MRM_feces/redo/" #select correct folder where the .txt files are 
META = "/media/sf_SF/Stage2021/Projects/MRM_feces/200805_MRM_Health_128_samples/200805_MRM_Health_128_samples_TM_corrected.csv" #path to metadata, make sure it is adapted with the script transform_metadata.py
EXCEL = "/media/sf_SF/Stage2021/Python_code_thermo_txt_to_ml/Projects/Lactic_acid/200929 MRM Health heranalyse trp lactic acid_Short.xls" #path to excel
PROJECT = "/media/sf_SF/Stage2021/Projects/" #path project directory where found/not found is(todo : change to parent dir PATH)

'''
############## Code ##############
##################################

### Part 1 : raw to txt ###
###########################
#see R-script
'''

### Part 2 : txt to df ### 
##########################
print("\nfrom text file to dataframe\n"+"-"*60)

#change path to the correct directory; 
os.chdir(PATH)

#loop over the directories and get the txt files
list_files = []
for file in files(PATH):
    if file.endswith(".txt") and not file.startswith("df_"): 
        list_files.append(file)
        txt_to_df(file, PATH)
        print("\tTransformed {}".format(file))
        #delete txt file? 

### Part 3 : df to png ### 
##########################

print("\nfrom dataframe to image\n"+"-"*60)

if not os.path.exists(PATH + "png"):
    os.mkdir(PATH+"png/")
for file in files(PATH):
    if file.startswith("df_") and file.endswith(".txt"):
        df_to_png(META, file, PATH)
        print("\tTransformed {}".format(file))

 
### Part 4 : png to ML ### 
##########################
print("\nfrom image to input data ML\n"+"-"*60)

##4.1 resize png's

if not os.path.exists(PATH + "resized"):
    os.mkdir(PATH+"resized/")
src_path = PATH + "png/"
dst_path = PATH + "resized/" 
resize_images(src_path, dst_path)

print("\tTransformed images\n")
    

##4.2 give label found/nf to png's 
if not os.path.exists(PROJECT + "found"):
    os.mkdir(PROJECT +"found/")
if not os.path.exists(PROJECT + "not_found"):
    os.mkdir(PROJECT +"not_found/")

#place the found/nf files manually in directory
