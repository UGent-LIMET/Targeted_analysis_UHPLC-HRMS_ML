# Explanation about the script 

#todo : replace found and not found by 1 and 0

############ Adjustments ############
#####################################

# project files
PATH = "/media/sf_SF/Stage2021/Projects/MRM_feces/y_matrix/" #select correct directory containing the excel files
NAMES = '/media/sf_SF/Stage2021/compare_names.xls'

############# Packages #############
####################################

import glob
import pandas as pd
import numpy as np
import xlwt

############## Code ##############
##################################

#input metadata
excel_files = glob.glob(PATH + "/*.xls")

#input the metabolite names for replacement
file_path = NAMES
df2 = pd.read_excel(file_path)
d = pd.Series(df2.original.values,index=df2.replacement).to_dict()
inv_map = {v: k for k, v in d.items()}

#loop over excel files + sheets 
for file in excel_files:
    wb = pd.ExcelFile(file)
    sheets = wb.book.sheets()
    total_df=pd.DataFrame()
    for sheet in sheets:
        try :
            if sheet.visibility == 0:
                df = wb.parse(sheet.name, skiprows=4)
                df = df[['Filename','Area','Sample ID']]
                df = df[df['Sample ID'].apply(lambda x: str(x).startswith("1"))] #sample id altijd 1? 
                mn = wb.parse(sheet.name, skiprows=1,nrows=1)
                mn = mn.iloc[0]['Component Name']
                df.insert(1, 'MetaboliteName', mn)
                df = df.iloc[:,:-1]
                df = df.replace({"MetaboliteName": inv_map})
                df['SampleTargetedcombination'] = df['Filename'] + "_" + df['MetaboliteName']
                df['Classification'] = np.where(df['Area'] == 'NF', 'NOT_FOUND', 'FOUND')
                df['Area']=df['Area'].replace({'NF':0})
                total_df = total_df.append(df, ignore_index = True).sort_values(by=['Filename'])
        
        except:
            print("there is an excel in the folder that should not be there") # change this (skip the file and go on)
    
    total_df.reset_index(drop=True, inplace= True)


    with pd.ExcelWriter(file + '_y_matrix.xls') as writer:
        total_df.to_excel(writer)
    
