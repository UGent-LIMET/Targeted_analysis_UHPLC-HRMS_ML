
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
PATH = "/media/sf_SF/Stage2021/Projects/MRM_feces/200805_MRM_Health_128_samples/" #select correct directory containing the bio, blank and std directories
META = "/media/sf_SF/Stage2021/Projects/MRM_feces/200805_MRM_Health_128_samples/200805_MRM_Health_128_samples_TM_corrected.csv" #path to metadata, make sure it is adapted with the script transform_metadata.py
EXCEL = "/media/sf_SF/Stage2021/Python_code_thermo_txt_to_ml/Projects/Lactic_acid/200929 MRM Health heranalyse trp lactic acid_Short.xls" #path to excel
PROJECT = "/media/sf_SF/Stage2021/Projects/" #path project directory where found/not found is(todo : change to parent dir PATH)

'''
# test files
PATH = "/media/sf_SF/Stage2021/Python_code_thermo_txt_to_ml/Projects_test/Lactic_acid/" #select correct directory containing the bio, blank and std directories
META = "/media/sf_SF/Stage2021/Python_code_thermo_txt_to_ml/Projects_test/Lactic_acid/200929_heranalyse_trp_lactic_acid_TM_corrected.csv" #path to metadata 
EXCEL = "/media/sf_SF/Stage2021/Python_code_thermo_txt_to_ml/Projects_test/Lactic_acid/200929 MRM Health heranalyse trp lactic acid_Short.xls" #path to excel
PROJECT = "/media/sf_SF/Stage2021/Python_code_thermo_txt_to_ml/Projects_test/" #path project directory (todo : change to parent dir PATH)
'''
############## Code ##############
##################################

### Part 1 : raw to txt ###
###########################
#see R-script


### Part 2 : txt to df ### 
##########################
print("\nfrom text file to dataframe\n"+"-"*60)

#change path to the correct directory; containing blank, bio, std
os.chdir(PATH)

#loop over the directories and get the txt files
list_files = []
for directory in directories(PATH):
    path = PATH+directory+"/"
    for file in files(path):
        if file.endswith(".txt") and not file.startswith("df_"): 
            list_files.append(file)
            txt_to_df(file, path)
            print("\tTransformed {} from {}".format(file,directory))
            #delete txt file? 


### Part 3 : df to png ### 
##########################

print("\nfrom dataframe to image\n"+"-"*60)


for directory in directories(PATH):
    path = PATH+directory+"/"
    if not os.path.exists(path + "png"):
        os.mkdir(path+"png/")
    for file in files(path):
        if file.startswith("df_") and file.endswith(".txt"):
            df_to_png(META, file, path)
            print("\tTransformed {} from {}".format(file, directory))

 
### Part 4 : png to ML ### 
##########################
print("\nfrom image to input data ML\n"+"-"*60)

##4.1 resize png's
for directory in directories(PATH):
    path = PATH+directory+"/"
    if not os.path.exists(path + "resized"):
        os.mkdir(path+"resized/")
    src_path = path + "png/"
    dst_path = path + "resized/" 
    resize_images(src_path, dst_path)
    
    print("\tTransformed images from {}\n".format(directory))
    

##4.2 give label found/nf to png's 
if not os.path.exists(PROJECT + "found"):
    os.mkdir(PROJECT +"found/")
if not os.path.exists(PROJECT + "not_found"):
    os.mkdir(PROJECT +"not_found/")

#place the found/nf files manually in directory
