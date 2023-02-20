
############Adjustments##############

#options
PATH = '/media/sf_SF/Stage2021/targetedQE/' 
OPTION = 'area' # area or label

## Adjustments

filename_Y_labels = 'total_y_matrix.txt'

########################



# load libraries
import pandas as pd
import os
import random
from PIL import Image
import numpy as np


#set paths
path_data_in = PATH + 'data/input/' + 'MachineLearning/'
path_data_out = PATH + 'data/output/' + 'MachineLearning/'
path_data_X = path_data_in + 'Xarrays_test/' #png's
path_data_y = path_data_in + 'Yarrays/' #labels


filename = path_data_y + filename_Y_labels

y = pd.read_csv(filename, sep = "\t")

filenames_X_train = []
filenames_X_test = []
directory_list = os.listdir(path_data_X)
random.shuffle(directory_list)
os.chdir(path_data_X)

i = 0
for filename in directory_list:
    if ".png" in filename :
        if i % 3 == 0: 
            filenames_X_test.append(path_data_X + filename)
        else:
            filenames_X_train.append(path_data_X + filename)
        i = i + 1
    
def load_X_if_matched_in_y(filenames_list, y):
    all_images_as_array=[]
    label=[] 
    area = []   

    for filename in filenames_list:
        filename_wopath = filename.split('Xarrays_test/')[1]
        filename_wopath = filename_wopath.strip(".png")

        matching_y = y[y.SampleTargetedcombination==filename_wopath]
        if len(matching_y) == 1:
            label.append(matching_y.iloc[0,5]) 
            area.append(matching_y.iloc[0,3])
        
            img=Image.open(filename)
            np_array = np.asarray(img)

            l,b,c = np_array.shape    
            np_array = np_array.reshape(l*b*c,)   
            all_images_as_array.append(np_array)

            
        if len(matching_y) != 1:
            continue
        
    if OPTION == 'area':
        option = area
    else:
        option = label       
    return np.array(all_images_as_array), np.array(option)
    
X_train,y_train = load_X_if_matched_in_y(filenames_X_train, y)
X_test, y_test = load_X_if_matched_in_y(filenames_X_test, y)

print(y_train)
print(y_test)