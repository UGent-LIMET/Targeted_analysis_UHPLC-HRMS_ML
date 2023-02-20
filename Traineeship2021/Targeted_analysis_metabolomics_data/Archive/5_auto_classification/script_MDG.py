# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:26:44 2021

@author: Marilyn
"""
#info voor loes: structuur R pipeline is
#> Data -> Input, Output, Tests
#> R_scipts
#> py_scripts
#onder data/input/EXPEMENT staat alles dat bij 1 proef hoort, vb map 'TEST_targetedQE' (zie inhoud onedrive)
#indiv data voor aanmaak x, y zit in map input/EXP/ in verschillende submappen vb bio (maar niet erg)
#voor uiteindelijke database (alle x,y), als in 1 experiment map zitten met 2 subfolders: Xarrays, Yarrays
#in testvoorbeeld zal ik zelfde folder gebruiken voor gemak



############Adjustments##############

#options
PATH_DI06C001 = 'C:/Users/Marilyn/Documents/Files/Werk_LCA/Pipeline_metabolomics/' 

## Adjustments
EXPERIMENT = 'TEST_targetedQE'

path = PATH_DI06C001

filename_Y_labels = 'Y_temp.txt'

########################



# load libraries
import pandas as pd
import os
import random
from PIL import Image
import numpy as np


#set paths
path_data_in = path + 'Data/input/' + EXPERIMENT + '/'
path_data_out = path + 'Data/output/' + EXPERIMENT + '/'
path_data_X = path_data_in + 'Xarrays/' #png's
path_data_y = path_data_in + 'Yarrays/' #labels



## Y
#loas all Y labels together
filename = path_data_y + filename_Y_labels
y = pd.read_csv(filename, sep='\t')


## X
#list all X files and devide in train OR test folder
filenames_X_train = []
filenames_X_test = []
directory_list = os.listdir(path_data_X)

#random order list with filenames
random.shuffle(directory_list)
    
i = 0
for filename in directory_list:
    #print (filename) #all files, folders
    #print (i)
    if ".png" in filename:
        #print (filename)
        if i % 3 == 0: 
            #1/3th of data is test set, rest in train
            #print(i)
            filenames_X_test.append(path_data_X + filename)
        else:
            filenames_X_train.append(path_data_X + filename)
        i = i + 1
        
 #check ok? 70-30 devide train - test? ok     
len(filenames_X_train)
len(filenames_X_test)


## load X data + Merge per train/test X's with Y to S1
#keep only non unique values


def load_X_if_matched_in_y(filenames_list, y):
    all_images_as_array=[]
    label=[]    

    for filename in filenames_list:
        #print(filename)
        #filename = filenames_X_train[0]
        filename_wopath = filename.split('Xarrays/')[1]
        #filename_wopath = filename_wopath[:-4] #wo .png todo, see same x/y !!!
    
        matching_y = y[y.Name==filename_wopath]
        if len(matching_y) == 1:
            label.append(matching_y.iloc[0,1]) #1st elem contains string NF/FOUND
            
            #load figure correctly as array [[], [], []]]
            img=Image.open(filename)
            np_array = np.asarray(img)
            l,b,c = np_array.shape    
            np_array = np_array.reshape(l*b*c,)   
            all_images_as_array.append(np_array)
            
        if len(matching_y) != 1:
            print("no or multiple match(es) in y found for: " + filename)
            continue
        
    return np.array(all_images_as_array), np.array(label)

#if re.match(filename_wopath, y.Name[0]): #todo search in volled colom, ev niet via regress want wo .png moet volled zelfde
        


X_train,y_train = load_X_if_matched_in_y(filenames_X_train, y)
X_test, y_test = load_X_if_matched_in_y(filenames_X_test, y)


## ML train
#model.filt()


## ML evaluate
#model.score()



