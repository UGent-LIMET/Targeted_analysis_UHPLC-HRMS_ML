# This script creates PNG's from the .TXT files.

# To run this script, the configuration file (with the wanted EXPERIMENT)
# and the METADATA (corrected with transform_metadata.py) are needed.

# The configuration file is always located in the parent directory of your experiment

# The created png's will be in the subfolder 'png' in the parent directory of your experiment
# (! If this folder already exist, your png's will be added !)


############ Adjustments ############
#####################################

# project files
CONFIG = "E:/targetedQE/data/input/Configuration.R" # path to the configuration file
                                                    # EXPERIMENT in the configuration file contains the name of the experiment folder you want to use
META = "E:/targetedQE/data/input/0.testfiles/test_uniform_scalebar_png/200929_heranalyse_trp_lactic_acid_TM_corrected.txt"  # path to metadata (.txt)
                                                                                                # make sure it is corrected with the script transform_metadata.py

#PATH = "C:/Users/Loes/OneDrive - Hogeschool West-Vlaanderen/SF/Stage2021/Pycharm/test_png_color_and_scale/"  # select correct directory containing the bio, blank and std directories

# Do not adjust below this point !

############# Packages #############
####################################

# import pandas as pd
import os
# from pyteomics import mzxml
from get_functions import files, directories
import numpy as np
import pandas as pd
import matplotlib as m
import matplotlib.pyplot as plt
import re


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
        PATH = CONFIG.rsplit("/", 1)[0]+"/"+experiment.replace("'","")+"/"



### Part 3 : df to png ###
##########################

print("\nfrom dataframe to image\n" + "-" * 60)

def df_to_png(META, file, path):
    # import metadata
    df_meta = pd.read_csv(META, sep='\t')
    # df_meta = open(META, "r")
    # print(df_meta)

    # iterate over metadata
    for index, row in df_meta.iterrows():
        # met_id = row[1]  # save id of metabolite
        metabolite_name = row[2]  # save name of metabolite
        ionisation_mode = row[3]  # save ionisation mode of metabolite
        MZ = str(row[4])  # save m/z of metabolite

        # remove unneccesary characters
        metabolite_name = re.sub('[\W_]+', '', metabolite_name)

        # input dataframe
        filename = path + file
        df_file = pd.read_csv(filename, sep='\t')

        # search scannumbers and only retain 1 ionisation mode (from metadata)
        ##positive mode = odd -> 1, negative = even -> 0
        if ionisation_mode == '-':
            df_file = df_file[df_file["scannumber"] % 2 == 0]
        if ionisation_mode == '+':
            df_file = df_file[df_file["scannumber"] % 2 == 1]

        ### make block : mz keep +- 0.2 Da arround center
        standard_down = float(MZ) - 0.02
        standard_up = float(MZ) + 0.02
        # print(standard_up, standard_down)

        #
        roi = df_file[(df_file["m_over_z"] >= standard_down) & (df_file["m_over_z"] <= standard_up)]
        roi = roi.round({'m_over_z': 4, 'time': 2})

        # order from low to high rt
        roi = roi.sort_values('time')
        # print(roi.intensity)

        # adapt name
        # size = len(file)
        # name = file[:size - 4]

        new_filename = file.strip(".txt")
        new_filename = new_filename.strip("df_")

        # plot + log without axis (for ML)
        plt.figure()
        plt.scatter(
            x=roi.time,
            y=roi.m_over_z,
            marker='.',
            c=roi.intensity,
            alpha=0.5,
            cmap = plt.cm.binary
        )
        plt.colorbar()
        plt.clim(0,5.75*10**9)
        plt.ylabel("MZ (Da)")
        plt.xlabel("Time (min)")
        plt.savefig(path + "png/"+ new_filename + '_test_scalebar.png')
        # close files

for directory in directories(PATH):
    path = PATH + directory + "/"
    if not os.path.exists(path + "png"):
        os.mkdir(path + "png/")
    for file in files(path):
        if file.startswith("df_") and file.endswith(".txt"):
            df_to_png(META, file, path)
            print("\tTransformed {} from {}".format(file, directory))

