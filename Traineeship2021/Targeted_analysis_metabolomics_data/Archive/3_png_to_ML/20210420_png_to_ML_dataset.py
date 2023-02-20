#!/usr/bin/python3

#import module
from resize_multiple_images import resize_multiple_images
from rename_multiple_files_test import rename_multiple_files
from get_data_test import get_data
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import re
import xlrd
import shutil
import numpy as np

os.chdir("/media/sf_SF/Stage2021/Python_code_thermo_txt_to_ml/Projects/Lactic_acid/bio/png")
cwd = os.getcwd() + "/"
'''
os.mkdir(cwd + "resized/") 

##resize 
#set source and destination path for the resized images
src_path = cwd
dst_path = cwd + "resized/" 
resize_multiple_images(src_path, dst_path)

#####################################################################################
##move pngs is correct map found or nf based on the excel
# todo : make function (read_excel.py)

#input metadata
parent_dir = os.pardir
analysis = parent_dir + "/" + parent_dir + "/200929 MRM Health heranalyse trp lactic acid_Short.xls" #metadata has to be in parent directory

wb = xlrd.open_workbook(analysis)
sheet = wb.sheet_by_index(2) #todo : search for correct tab

file_name = sheet.col_values(0)
found_nf = sheet.col_values(6)
data = {'file_name':file_name, 'f/nf':found_nf}
df = pd.DataFrame.from_dict(data)
df = df.drop(index= [0,1,2,3,4])
# print(df)

#make directory for found and nf 
os.mkdir(cwd + "found/")
os.mkdir(cwd + "nf/")

#get all the png's 
all_files = os.listdir(cwd)
list_names = []
list_files = []
for filename in all_files:
    if ".png" and "Lacticacid" in filename: #search for png files : here lactic acid
        file_split = re.split("_|\.", filename)
        file_no = file_split[1]
        list_names.append(file_no)
        list_files.append(filename)
# list_names.append("200929s052") #to test 
# list_names.append("200929s037") #to test 
# print(list_names)
# print(list_files)

match =  df.loc[df['file_name'].isin(list_names)]
fnf =  match.loc[:, 'f/nf']

for index, row in match.iterrows():
    if row['f/nf'] == "NF":
        fn = row['file_name']
        print("NOT FOUND =  " + fn)
        search_pat = "df_" + fn
        # print(search_pat)

        match_file = list(filter(lambda x: x.startswith(search_pat), list_files))
        print(match_file)
        if not match_file:
            # empty 
            continue
        else :
            listToStr = ' '.join([str(elem) for elem in match_file])
            print(listToStr)
            nf_source = cwd + "resized/" + listToStr
            nf_dest = cwd + "nf/" + listToStr
            print(nf_source)
            print(nf_dest)
            shutil.copy2(nf_source, nf_dest) #copy to test, use move
            print("\n")
        
        #move file to nf map
        
    else:
        fn = row['file_name']
        print("FOUND =  " + fn)
        search_pat = "df_" + fn
        # print(search_pat)

        match_file = list(filter(lambda x: x.startswith(search_pat), list_files))
        print(match_file)
        if not match_file:
            # empty 
            continue
        else :
            listToStr = ' '.join([str(elem) for elem in match_file])
            print(listToStr)
            f_source = cwd + "resized/" + listToStr
            f_dest = cwd + "found/" + listToStr
            print(f_source)
            print(f_dest)
            shutil.copy2(f_source, f_dest) #move
            print("\n")

        #move file to found map


#####################################################################################

##rename 
#set path and obj : found
#path= dst_path
path_f=cwd + "found_dop/"
obj= "_FOUND"
print(path_f)
rename_multiple_files(path_f,obj)

#set path and obj : nf
# path= dst_path
path_nf=cwd + "nf_dop/"
obj= "_NF"
print(path_nf)
rename_files(path_nf,obj)
'''
##move images to test and train map 

##get data
#set path
path_to_train_set = "/media/sf_SF/Stage2021/Python_code_thermo_txt_to_ml/Projects/Lactic_acid/train_all/" 
path_to_test_set = "/media/sf_SF/Stage2021/Python_code_thermo_txt_to_ml/Projects/Lactic_acid/test_all/" 
#train and test data
X_train,y_train = get_data(path_to_train_set)
X_test, y_test = get_data(path_to_test_set)


##test
# print('X_train set : ',X_train) 
# print('y_train set : ',y_train) 
# print('X_test set : ',X_test)
# print('y_test set : ',y_test)

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)


##SVM
print('\nSVM')
print('-'*80)
# Create a classifier: a support vector classifier

classifier = svm.SVC(gamma=0.001, kernel='rbf')

# We learn the digits on the first half of the digits
classifier.fit(X_train, y_train)

# Now predict the value of the digit on the second half:
predicted = classifier.predict(X_test)
print("predicted = {}".format(predicted))
#real answer test label + accuracy score
print("test set  = {}".format(y_test))
acc = accuracy_score(y_test, predicted)
print("accuracy  = {}".format(acc))

##Naive Bayes
print('\nNaive Bayes')
print('-'*80)
nv = GaussianNB() # create a classifier
nv.fit(X_train,y_train) # fitting the data
y_pred = nv.predict(X_test) # store the prediction data
print("predicted = {}".format(y_pred))
#real answer test label + accuracy score
print("test set  = {}".format(y_test))
acc = accuracy_score(y_test,y_pred) # calculate the accuracy
print("accuracy  = {}\n".format(acc))

'''
##Neural Network  todo : preprocess 
print('\nNeural Network')
print('-'*80)
clf = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu').fit(X_train, y_train)
clf.predict_proba(X_test)
predicted = clf.predict(X_test)
print("predicted = {}".format(predicted))
#real answer test label + accuracy score
print("test set  = {}".format(y_test))
score = clf.score(X_test, y_test)
print("accuracy  = {}\n".format(score))
'''
##Random Forest 
print('\nRandom Forest')
print('-'*80)
rf = RandomForestClassifier(max_depth = 2, random_state = 0)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
print("predicted = {}".format(predictions))
print("test set  = {}".format(y_test))
acc_score = rf.score(X_test, y_test)
print("accuracy  = {}\n".format(acc_score))
