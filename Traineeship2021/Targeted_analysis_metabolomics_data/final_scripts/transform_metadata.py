# Title: transform_metadata.py
# Owner: Laboratory of Chemical Analysis (LCA), Ghent University
# Creator: Loes Vervaele
# Running title: R pipeline
# Script: Part I bis: Targeted Analysis QE

# This script adjusts the METADATA.
# The correct columns (ID, Ionisation mode, M/Z and retention time) are selected.
# The wanted output is written to a TXT file.
# The new file has the original name with '_corrected.txt' added.
# Eventually there can be white lines in the .txt file, they have to be removed.

############# Packages #############
####################################

import pandas as pd

############ Adjustments ############
#####################################

META = "" # path to metadata, make sure it is adapted with the script transform_metadata.py

# ----- Do not adjust below this point! ----- 

############## Code ##############
##################################

# import metadata
df_meta = pd.read_csv(META, sep='\t', encoding='ISO-8859-1')
# select the correct columns
df_cols = ["ID", "Metabolite", "Ionisation mode", "Emperical m/z (Da)", "Retention time (min)"]
df_meta = df_meta.loc[:, df_cols]

# rename the columns (for compatibility with existing R-scripts)
df_meta = df_meta.rename(
    columns={"Metabolite": "MetaboliteName", "Ionisation mode": "IonisationMode", "Emperical m/z (Da)": "MZ",
             "Retention time (min)": "Time"})

# export df to .csv
csv_name = META.split(".txt")[0] + "_corrected.txt"
df_meta.to_csv(csv_name, sep='\t')

# eventually TODO: delete all white lines in the created txt file with the script
