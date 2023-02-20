
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
PROJECT = "/media/sf_SF/Stage2021/Projects_ML_test/" #path directory containing found and not_found 

############## Code ##############
##################################
##4.3 rename png's
'''
#found
path_f = PROJECT + "found/"
obj= "_FOUND"
rename_files(path_f,obj)
#not_found
path_nf = PROJECT + "not_found/"
obj= "_NF"
rename_files(path_nf,obj)

print("\tRenamed the images\n")

##4.3 place all the png's in one map

if not os.path.exists(PROJECT + "ML_all"):
    os.mkdir(PROJECT + "ML_all/")

founds = os.listdir(PROJECT+"found/")
not_founds = os.listdir(PROJECT+"not_found/")

for png in founds:
    shutil.copy(os.path.join(PROJECT+"found/",png),PROJECT+"ML_all/")
for png in not_founds:
    shutil.copy(os.path.join(PROJECT+"not_found/",png),PROJECT+"ML_all/")


print("\tMoved the images\n")

#4.4 divide the png's over test and train data (2/3 train, 1/3 test)

if not os.path.exists(PROJECT + "ML_train"):
    os.mkdir(PROJECT +"ML_train/")
if not os.path.exists(PROJECT + "ML_test"):
    os.mkdir(PROJECT +"ML_test/")

total = len([img for img in os.listdir(PROJECT+"ML_all")])
num_test = round(total*(1/3))
num_train = round(total*(2/3))

random_test = random.sample(os.listdir(PROJECT+"ML_all"), num_test)
random_train = random.sample(os.listdir(PROJECT+"ML_all"), num_train)

for test_data in random_test:
    shutil.copy(os.path.join(PROJECT+"ML_all/",test_data),PROJECT+"ML_test/")
for train_data in random_train:
    shutil.copy(os.path.join(PROJECT+"ML_all/",train_data),PROJECT+"ML_train/")

print("\tDivided the images\n")

#4.4.1 Divide de test data in random groups
os.chdir(PROJECT+"ML_test/")
print(os.listdir(os.getcwd()))

'''
#4.5 get data
X_train,y_train = get_data(PROJECT+"ML_train/")
X_test, y_test = get_data(PROJECT+"ML_test/")



print("\tGot the data from the images\n")
'''
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
'''
print('\nRandom Forest')
print('-'*30)
RF(X_train, y_train, X_test, y_test)
