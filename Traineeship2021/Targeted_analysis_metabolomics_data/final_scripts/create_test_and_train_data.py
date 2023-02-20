
# Title: create_test_and_train_data.py
# Owner: Laboratory of Chemical Analysis (LCA), Ghent University
# Creator: Loes Vervaele
# Running title: R pipeline
# Script: Part I bis: Targeted Analysis QE

# This script creates test and train data for Machine Learning purposes.

# To run this script, you need the folder containing the resized png's
# and the training labels.

# Choose the wanted option, label for classification or area for regression.

############Adjustments##############


PATH = '/media/sf_SF/Stage2021/targetedQE/' # directory containing all the data
OPTION = 'label'  # area or label (classification => label, for regression => area)
FOLDER = 'X_arrays/' # folder containing the resized png's
filename_Y_labels = 'total_y_matrix.txt' # txt file with the training labels

# Do not adjust below this point !

############# Packages #############
####################################

import pandas as pd
import os
import random
from PIL import Image
import numpy as np

############## Code ##############
##################################
# set paths
path_data_in = PATH + 'data/input/' + 'MachineLearning/'
path_data_out = PATH + 'data/output/' + 'MachineLearning/'
path_data_X = path_data_in + FOLDER  # png's
path_data_y = path_data_in + 'Yarrays/'  # labels

filename = path_data_y + filename_Y_labels

y = pd.read_csv(filename, sep="\t")

filenames_X_train = []
filenames_X_test = []
directory_list = os.listdir(path_data_X)
random.shuffle(directory_list)
os.chdir(path_data_X)

# divide data in test and train set
i = 0
for filename in directory_list:
    if ".png" in filename:
        if i % 3 == 0:
            filenames_X_test.append(path_data_X + filename)
        else:
            filenames_X_train.append(path_data_X + filename)
        i = i + 1


def load_X_if_matched_in_y(filenames_list, y):
    all_images_as_array = []
    label = []
    area = []

    ordered_filenames_lab = pd.DataFrame()
    ordered_filenames_ar = pd.DataFrame()

    for filename in filenames_list:
        filename_wopath = filename.split(FOLDER)[1]
        filename_wopath = filename_wopath.strip(".png")

        matching_y = y[y.SampleTargetedcombination == filename_wopath]  # search for a match
        if len(matching_y) == 1:  # use the png if there is exact 1 match
            lab = matching_y.iloc[0, 5]  # save the label found/not found
            label.append(lab)
            ar = matching_y.iloc[0, 3]  # save the area
            area.append(ar)

            img = Image.open(filename)
            np_array = np.asarray(img)

            l, b = np_array.shape
            np_array = np_array.reshape(l * b, )
            all_images_as_array.append(np_array)

            file_area = pd.DataFrame(np.array([[filename_wopath, ar]]))  # save the filename together with the area
            ordered_filenames_ar = ordered_filenames_ar.append(file_area, ignore_index=True)

            file_label = pd.DataFrame(np.array([[filename_wopath, lab]]))  # save the filename together with the label
            ordered_filenames_lab = ordered_filenames_lab.append(file_label, ignore_index=True)

        if len(matching_y) != 1:
            continue

    # this is extra, to check the used files
    if OPTION == 'area':
        option = area
        df = ordered_filenames_ar
    else:
        option = label
        df = ordered_filenames_lab
    return np.array(all_images_as_array), np.array(option), df


X_train, y_train, df_train = load_X_if_matched_in_y(filenames_X_train, y)
X_test, y_test, df_test = load_X_if_matched_in_y(filenames_X_test, y)

