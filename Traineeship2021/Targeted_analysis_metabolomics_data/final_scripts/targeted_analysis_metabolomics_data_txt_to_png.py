# Title: targeted_analysis_metabolomics_data_txt_to_png.py
# Owner: Laboratory of Chemical Analysis (LCA), Ghent University
# Creator: Dr. Marilyn De Graeve; Loes Vervaele
# Running title: R pipeline
# Script: Part I bis: Targeted Analysis QE

# This script creates PNG's from the .TXT files.

# To run this script, the configuration file (with the wanted EXPERIMENT)
# and the METADATA (corrected with transform_metadata.py) are needed.

# The configuration file is always located in the parent directory of your experiment

# The created png's will be in the subfolder 'png' in the directory of your experiment
# The resized png's will be in the subfolder 'resized' in the parent directory of your experiment
# (! If this folder already exist, your resized png's will be added !)


############ Adjustments ############
#####################################

# project files
CONFIG = "E:/targetedQE/data/input/Configuration.R" # path to the configuration file
    # EXPERIMENT in the configuration file contains the name of the experiment folder you want to use
META = "E:/targetedQE/data/input/0.testfiles/mzXML_or_txt_to_png/200423_72samples_succinate_TM_corrected.txt"  # path to metadata (.txt)
    # make sure it is corrected with the script transform_metadata.py

#------ Do not adjust below this point ! ------

############# Packages #############
####################################

# import pandas as pd
import os
# from pyteomics import mzxml
from get_functions import files, directories
from transform_functions import txt_to_df, df_to_png, resize_images

############## Code ##############
##################################

### Part 1 : read path in config file ###
#########################################

config_file = open(CONFIG, 'r')
Lines = config_file.readlines()

# Strips the newline character
for line in Lines:
    if line.startswith('EXPERIMENT'):
        experiment = line.split(' <- ')
        experiment = experiment[1].split(' #')[0]
        PATH = CONFIG.rsplit("/", 1)[0] + "/" + experiment.replace("'", "") + "/"

### Part 2 : txt to df ###
###########################

print("\nfrom txt file to dataframe\n" + "-" * 60)

# change path to the correct directory; containing blank, bio, std
os.chdir(PATH)

# loop over the directories and get the txt files
list_files = []
for directory in directories(PATH):
    path = PATH + directory + "/"
    for file in files(path):
        if file.endswith(".txt") and not file.startswith("df_"):
            list_files.append(file)
            txt_to_df(file, path)
            print("\tTransformed {} from {}".format(file, directory))

### Part 3 : df to png ###
##########################

print("\nfrom dataframe to image\n" + "-" * 60)

for directory in directories(PATH):
    path = PATH + directory + "/"
    if not os.path.exists(path + "png/"):
        os.mkdir(path + "png/")
    for file in files(path):
        if file.startswith("df_") and file.endswith(".txt"):
            df_to_png(META, file, path)
            print("\tTransformed {} from {}".format(file, directory))

### Part 4 : png to ML ###
##########################
print("\nfrom image to input data ML\n" + "-" * 60)

##4.1 resize png's
for directory in directories(PATH):
    path = PATH + directory + "/"
    print(PATH)
    print(path)

    if not os.path.exists(PATH + "resized/"):
        os.mkdir(PATH + "resized/")
    src_path = path + "png/"
    dst_path = PATH + "resized/"
    resize_images(src_path, dst_path)


print("\tTransformed images from {}\n".format(directory))
