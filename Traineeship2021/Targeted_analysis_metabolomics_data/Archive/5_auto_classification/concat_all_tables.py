
# Explanation about the script 


############ Adjustments ############
#####################################

# project files
PATH = "/media/sf_SF/Stage2021/Projects/MRM_feces/y_matrix/" #select correct parent directory containing the project folders where all the excel files are


############# Packages #############
####################################

import glob
import pandas as pd
import numpy as np
import xlwt
import os
# import openpyxl


############## Code ##############
##################################

os.chdir(PATH)
df= pd.DataFrame()
for (dirpath, dirnames, filenames) in os.walk(PATH):
    for filename in filenames:
        if filename.endswith('_y_matrix.txt'): 
            path_excel = os.sep.join([dirpath, filename])
            df = df.append(pd.read_csv(path_excel, sep =","), ignore_index=True) 

df = df.drop('Unnamed: 0', axis=1)

file_name = 'total_y_matrix_TEST.txt'
file_exists = os.path.isfile(file_name) 
if file_exists:
    print("The file already exists.")
else:
    df.to_csv(file_name, sep="\t")

