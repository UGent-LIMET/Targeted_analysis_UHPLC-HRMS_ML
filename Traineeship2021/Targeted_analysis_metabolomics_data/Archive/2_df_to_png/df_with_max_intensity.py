# This script creates a file with the maximum intensity per dataframe
############ Adjustments ############
#####################################

# project files
PATH = "E:/targetedQE/data/input/0.testfiles/all_df/" # path to folder with dataframes                                                                                          # make sure it is corrected with the script transform_metadata.py

# Do not adjust below this point !

############# Packages #############
####################################

import pandas as pd
import os
import csv
import json
# from pyteomics import mzxml
from get_functions import files, directories
from transform_functions import txt_to_df, df_to_png, resize_images

############## Code ##############
##################################

files = os.listdir(PATH)
list = []
for file in files :
    if file.startswith("df_"):
        df = pd.read_csv(PATH + file, sep="\t")
        # print(df)
        max_value = df["intensity"].max()
        # print("The maximum intensity of " + file + " is : ")
        # print(max_value)
        d = {"file" : file, "max_value" : max_value}
        # print(d)
        list.append(d)

# print(list)
with open('max_intensity_per_df.txt','w', newline="") as file:
    fc = csv.DictWriter(file,
                        fieldnames=list[0].keys())
    fc.writerows(list)