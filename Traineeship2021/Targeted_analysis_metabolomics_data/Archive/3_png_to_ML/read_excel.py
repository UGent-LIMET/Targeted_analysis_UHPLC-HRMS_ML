#!/usr/bin/python3

#packages
import pandas as pd
import os
import re
import xlrd
import shutil

##function move png's to correct map
def Move_png_based_on_excel()
os.chdir("/media/sf_SF/Stage2021/Python_code_thermo_txt_to_ml/Testfiles/png/") #select correct directory / todo : change for pipeline
cwd = os.getcwd() + "/"

all_files = os.listdir(cwd)

list_names = []
for filename in all_files:
    if ".png" and "Lacticacid" in filename: #search for png files : here lactic acid
        file_split = re.split("_|\.", filename)
        file_no = file_split[1]
        list_names.append(file_no)
list_names.append("200929s052") #to test 
print(list_names)
'''        
os.mkdir(cwd + "found/")
os.mkdir(cwd + "nf/")
'''


#input metadata
parent_dir = os.pardir
analysis = parent_dir + "/" + parent_dir + "/200929 MRM Health heranalyse trp lactic acid_Short.xls" #metadata has to be in parent directory

wb = xlrd.open_workbook(analysis)
sheet = wb.sheet_by_index(2) #todo : search for correct tab

file_name = sheet.col_values(0)
found_nf = sheet.col_values(6)

d= {"filename":file_name,'found/nf':found_nf}
df = pd.DataFrame(d, columns=["filename","found/nf"])
df = df.drop([0,1,2,3,4]) # todo : remove correct rows for all files

for item in list_names :
    if item in list(df['filename']):
        print("match")
        # shutil.move()
    else :
        print("no match")

