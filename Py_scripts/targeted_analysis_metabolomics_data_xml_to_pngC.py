# Title: targeted_analysis_metabolomics_data_xml_to_png.py
# Owner: Laboratory of Integrative Metabolomics (LIMET), Ghent University
# Creator: Dr. Marilyn De Graeve; Loes Vervaele
# Maintainer: <limet@ugent.be>
# Script: Part I bis: Targeted Analysis QE

# This script creates PNG's from the .mzXML files.

# To run this script, the configuration file (with the wanted EXPERIMENT)
# and the METADATA (corrected with transform_metadata.py) are needed.

# The configuration file is always located in the parent directory of your experiment

# The created png's will be in the subfolder 'png' in the directory of your experiment
# The resized png's will be in the subfolder 'resized' in the parent directory of your experiment
# (! If this folder already exist, your resized png's will be added !)

############ Adjustments ############
#####################################

# project files
CONFIG = "XXX/Configuration.R" # path to the configuration file
    # EXPERIMENT in the configuration file contains the name of the experiment folder you want to use
META = "XXX/TM.txt"  # path to metadata (.txt)
    # make sure it is corrected with the script transform_metadata.py

# Do not adjust below this point !

############# Packages #############
####################################

import pandas as pd
import os
from pyteomics import mzxml
from get_functions import files, directories
from transform_functions import df_to_png, resize_images
from multiprocessing import Process, freeze_support

############## Code ##############
##################################

### Part 1 : read path in config file ###
#########################################
def f():
    config_file = open(CONFIG, 'r')
    Lines = config_file.readlines()

    # Strips the newline character
    for line in Lines:
        if line.startswith('EXPERIMENT'):
            experiment = line.split(' <- ')
            experiment = experiment[1].split(' #')[0]
            PATH = CONFIG.rsplit("/", 1)[0]+"/"+experiment.replace("'","")+"/"



    ### Part 2 : xml to df ###
    ###########################

    print("\nfrom xml file to dataframe\n" + "-" * 60)

    # check for correct xml files in wd
    for directory in directories(PATH):
        path = PATH + directory + "/"
        os.chdir(path)
        for filename in files(path):
            if filename.endswith(".mzXML"):  # TODO : write in function
                df = pd.DataFrame()
                df_total = pd.DataFrame()

                f = mzxml.MzXML(filename)
                for spec in f.map():
                    num = (spec['num'])
                    rt = (spec['retentionTime'])
                    mz = (spec['m/z array'])
                    intensity = (spec['intensity array'])

                    df = pd.DataFrame(data=[intensity, mz])
                    df = df.T

                    for row in df.iterrows():
                        df["number"] = num
                        df["rt"] = rt
                    df_total = df_total.append(df)

                df_cols = ["intensity", "m/z", "number", "time"]
                df_total.columns = df_cols
                df_total = df_total[["number", "time", "intensity", "m/z"]]
                df_total.to_csv('df_' + filename.strip('.mzXML') + '.txt', index=None, sep='\t')

    ### Part 3 : df to png ###
    ##########################

    print("\nfrom dataframe to image\n" + "-" * 60)

    # df_meta = pd.read_csv(META, sep='\t')

    for directory in directories(PATH):
        path = PATH + directory + "/"
        if not os.path.exists(path + "png"):
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
        if not os.path.exists(PATH + "resized"):
            os.mkdir(PATH + "resized/")
        src_path = path + "png/"
        dst_path = PATH + "resized/"
        resize_images(src_path, dst_path)

        print("\tTransformed images from {}\n".format(directory))

if __name__ == '__main__':
    freeze_support()
    Process(target=f).start()
