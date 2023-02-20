# goal : df with scannumber, time, intesity and m_over_z of mzXML files 


############# Packages #############
####################################

import os
import pandas as pd 
# from pyopenms import *
from pyteomics import mzxml, auxiliary
import os

############# Function #############
####################################

##function list directories
def directories(path):
    for directory in os.listdir(path):
        if os.path.isdir(os.path.join(path, directory)):
            yield directory

##function list files
def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file



############ Adjustments ############
#####################################
PATH = "/media/sf_SF/Stage2021/targetedQE/data/input/xml/" #directory where the bio/std/blank folders are
                                                           # containing the xml files


###########################################################################################################

#check for correct xml files in wd
# list_files = []
for directory in directories(PATH):
    path = PATH+directory+"/"
    os.chdir(path)
    for filename in files(path):
        if filename.endswith(".mzXML") : #TODO : write in function
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

            df_cols = ["intensity","m/z","number","time"]
            df_total.columns = df_cols
            df_total = df_total[["number", "time", "intensity", "m/z"]]
            df_total.to_csv('df_'+filename.strip('.mzXML')+'.txt', index=None, sep='\t') 


# print(list_files)
