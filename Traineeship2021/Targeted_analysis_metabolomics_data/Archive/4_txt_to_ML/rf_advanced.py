# Explanation about the script 

############# Packages #############
####################################

from random import sample
import os, shutil
from transform_functions import get_data
from sklearn.ensemble import RandomForestClassifier

############ Adjustments ############
#####################################
PROJECT = "/media/sf_SF/Stage2021/Python_code_thermo_txt_to_ml/Projects_ML_test/" #path directory containing found and not_found 
NUM_PNG = 50 #adjust how big the train data must be

############## Code ##############
##################################
'''
##4.3 place all the png's in one map

founds = os.listdir(PROJECT+"found/")
length_fo = len(founds)
not_founds = os.listdir(PROJECT+"not_found/")
length_nf = len(not_founds)
print("amount found = {}\namount not found = {}".format(length_fo, length_nf))

# select files for the test data (founds)
test_fo = (sample(founds,round(length_fo/3)))
for png in test_fo:
    shutil.copy(os.path.join(PROJECT+"found/",png),PROJECT+"ML_test/")
# delete the selection from the list
train_fo = list(set(founds) - set(test_fo))
for png in train_fo:
    shutil.copy(os.path.join(PROJECT+"found/",png),PROJECT+"ML_train/")
print("total = {}\ntest = {}\ntrain = {}\n".format(length_fo,len(test_fo),len(train_fo)))

# select files for the test data (not founds)
test_nf = (sample(not_founds,round(length_nf/3)))
for png in test_nf:
    shutil.copy(os.path.join(PROJECT+"not_found/",png),PROJECT+"ML_test/")
# delete the selection from the list
train_nf = list(set(not_founds) - set(test_nf))
for png in train_nf:
    shutil.copy(os.path.join(PROJECT+"not_found/",png),PROJECT+"ML_train/")
print("total = {}\ntest = {}\ntrain = {}\n".format(length_nf,len(test_nf),len(train_nf)))

print("\tDivided and moved the images\n")

'''
'''
## 4.4 make subgroups to train/test the data
#select subpart of the train data 
train_files = os.listdir(PROJECT+"ML_train/")
tr_founds = []
tr_nf = []
for png in train_files:
    if "FOUND" in png:
        tr_founds.append(png)
    elif "NF" in png:
        tr_nf.append(png)
    else:
        print("mistake")
sel_train = (sample(tr_founds,round(NUM_PNG/2)))+(sample(tr_nf,round(NUM_PNG/2)))
if not os.path.exists(PROJECT + "sel_train"):
    os.mkdir(PROJECT + "sel_train/")
for png in sel_train:
    shutil.copy(os.path.join(PROJECT+"ML_train/",png),PROJECT+"sel_train/")

#select subpart of the test data 
test_files = os.listdir(PROJECT+"ML_test/")
te_founds = []
te_nf = []
for png in test_files:
    if "FOUND" in png:
        te_founds.append(png)
    elif "NF" in png:
        te_nf.append(png)
    else:
        print("mistake")
sel_test = (sample(te_founds,round(NUM_PNG/4)))+(sample(te_nf,round(NUM_PNG/4)))
if not os.path.exists(PROJECT + "sel_test"):
    os.mkdir(PROJECT + "sel_test/")
for png in sel_test:
    shutil.copy(os.path.join(PROJECT+"ML_test/",png),PROJECT+"sel_test/")
'''
X_train,y_train = get_data(PROJECT+"sel_train/")
X_test, y_test = get_data(PROJECT+"sel_test/")

print('\nRandom Forest')
print('-'*30)
growing_rf = RandomForestClassifier(max_depth = 2, random_state = 0, n_estimators=10, n_jobs=-1, 
                                        warm_start=True)
for i in range(40):
    growing_rf.fit(X_train, y_train)
    growing_rf.n_estimators += 10
predictions = growing_rf.predict(X_test)
print("predicted = {}".format(predictions))
print("test set  = {}".format(y_test))
acc_score = growing_rf.score(X_test, y_test)
print("accuracy  = {}\n".format(acc_score))