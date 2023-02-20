
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
from ML_models_functions import SVM, NB, NN, RF

############ Adjustments ############
#####################################

#PATH = "/media/sf_SF/Stage2021/Python_code_thermo_txt_to_ml/Projects/Lactic_acid/" #select correct directory containing the bio, blank and std directories
PATH = "/media/sf_SF/Stage2021/Projects/" #test directory
'''
META = "/media/sf_SF/Stage2021/Python_code_thermo_txt_to_ml/Projects_test/Lactic_acid/200929_heranalyse_trp_lactic_acid_TM.txt" #path to metadata 
EXCEL = "/media/sf_SF/Stage2021/Python_code_thermo_txt_to_ml/Projects_test/Lactic_acid/200929 MRM Health heranalyse trp lactic acid_Short.xls" #path to excel
'''
############## Code ##############
##################################

### Part 1 : raw to txt ###
###########################
#see R-script


### Part 2 : txt to df ### 
##########################
'''
print("\nfrom text file to dataframe\n"+"-"*60)

#change path to the correct directory; containing blank, bio, std
os.chdir(PATH)

#loop over the directories and get the txt files
list_files = []
for directory in directories(PATH):
    path = PATH+directory+"/"
    for file in files(path):
        if file.endswith(".txt"):
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
#todo 
if not os.path.exists(PATH + "found"):
    os.mkdir(PATH +"found/")
if not os.path.exists(PATH + "not_found"):
    os.mkdir(PATH +"not_found/")


##4.3 rename png's
#found
path_f = PATH + "found/"
obj= "_FOUND"
rename_files(path_f,obj)
#not_found
path_nf = PATH + "not_found/"
obj= "_NF"
rename_files(path_nf,obj)

print("\tRenamed the images\n")
'''
##4.3 place all the png's in one map

if not os.path.exists(PATH + "ML_all"):
    os.mkdir(PATH + "ML_all/")

founds = os.listdir(PATH+"found/")
not_founds = os.listdir(PATH+"not_found/")

for png in founds:
    shutil.copy(os.path.join(PATH+"found/",png),PATH+"ML_all/")
for png in not_founds:
    shutil.copy(os.path.join(PATH+"not_found/",png),PATH+"ML_all/")
    
print("\tMoved the images\n")

#4.4 divide the png's over test and train data (2/3 train, 1/3 test)
if not os.path.exists(PATH + "ML_train"):
    os.mkdir(PATH +"ML_train/")
if not os.path.exists(PATH + "ML_test"):
    os.mkdir(PATH +"ML_test/")

total = len([img for img in os.listdir(PATH+"ML_all")])
num_test = round(total*(1/3))
num_train = round(total*(2/3))

random_test = random.sample(os.listdir(PATH+"ML_all"), num_test)
random_train = random.sample(os.listdir(PATH+"ML_all"), num_train)

for test_data in random_test:
    shutil.copy(os.path.join(PATH+"ML_all/",test_data),PATH+"ML_test/")
for train_data in random_train:
    shutil.copy(os.path.join(PATH+"ML_all/",train_data),PATH+"ML_train/")

print("\tDivided the images\n")

#4.5 get data
X_train,y_train = get_data(PATH+"ML_train/")
X_test, y_test = get_data(PATH+"ML_test/")

print("\tGot the data from the images\n")

#4.6 creating models 
print('\nSVM')
print('-'*30)
SVM(X_train, y_train, X_test, y_test)

print('\nNaive Bayes')
print('-'*30)
NB(X_train, y_train, X_test, y_test)

print('\nNeural Network')
print('-'*30)
NN(X_train, y_train, X_test, y_test)

print('\nRandom Forest')
print('-'*30)
RF(X_train, y_train, X_test, y_test)
