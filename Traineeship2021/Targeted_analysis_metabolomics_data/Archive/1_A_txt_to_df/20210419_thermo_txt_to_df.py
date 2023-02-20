#!/usr/bin/python3

#packages
import os
import ThermoTXT_to_LongDF as df_fct

#set wd
os.chdir("/media/sf_SF/Stage2021/Python_code_thermo_txt_to_ml/Projects/Lactic_acid/blank") #select correct directory 

#check for correct txt files in wd
directory_list = os.listdir(os.getcwd())
print(directory_list)
for filename in directory_list:
    if ".txt" in filename: #search for text files, raw files are not used
        print(filename)
        df_fct.ThermoTXT_to_LongDF(filename) #make df for each txt file


