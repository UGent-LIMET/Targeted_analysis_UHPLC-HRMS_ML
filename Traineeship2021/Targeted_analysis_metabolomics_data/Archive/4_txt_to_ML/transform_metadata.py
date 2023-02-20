
# Explanation about the script 

############# Packages #############
####################################

import pandas as pd
import chardet

############ Adjustments ############
#####################################
#META = "/media/sf_SF/Stage2021/Projects/MRM_feces/Lactic_acid/200929_heranalyse_trp_lactic_acid_TM.txt" #path to metadata 
META = "E:/targetedQE/data/input/Prodigest/180320_polair/180320_polair_TM.txt"
############## Code ##############
##################################
'''
with open(META, 'rb') as rawdata:
    result= chardet.detect(rawdata.read(100000))
print(result)

'''

#import metadata 
df_meta = pd.read_csv(META, sep='\t', encoding='ISO-8859-1')

#select the correct columns
df_cols = ["ID", "Metabolite", "Ionisation mode", "Emperical m/z (Da)", "Retention time (min)"]
df_meta = df_meta.loc[:, df_cols]

#rename the columns (for compatibility with existing R-scripts)
df_meta = df_meta.rename(columns={"Metabolite": "MetaboliteName", "Ionisation mode": "IonisationMode", "Emperical m/z (Da)" : "MZ", "Retention time (min)" : "Time" })
print(df_meta)
'''
#export df to .csv 
csv_name = META.split(".txt")[0]+"_corrected.csv"
df_meta.to_csv(csv_name)
'''
