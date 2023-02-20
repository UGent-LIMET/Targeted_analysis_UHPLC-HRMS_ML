
EXCEL = '/media/sf_SF/Stage2021/Projects/MRM_feces/y_matrix/total_y_matrix.txt'
PNG = '/media/sf_SF/Stage2021/all_png/'


import pandas as pd
import os
import re

# import de lijst met png's 
png_list = os.listdir(PNG)
# print(png_list)

# import de excel met de labels
df = pd.read_csv(EXCEL, sep="\t")
print(df)
# print(df.columns)
# target_df = df[['SampleTargetedcombination']]
# print(target_df)

target_list = df['SampleTargetedcombination'].tolist()
# print(target_list)

#counters 
no_match = 0
match = 0

# split de target name
for target in target_list:
    # print(target)
    split = target.split("_")
    
    total = split[0]+".+"+split[1]
    # print(total)
    
    # print(split)
    for png in png_list:
        if re.search(total, png):
            match = match + 1 
        else :
            no_match = no_match + 1

print("match : {}".format(match))
print("no match : {}".format(no_match))

