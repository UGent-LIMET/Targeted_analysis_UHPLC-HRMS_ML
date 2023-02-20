#!/usr/bin/python3

#packages
import numpy as np
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as m
import os
import re

os.chdir("/media/sf_SF/Stage2021/Python_code_thermo_txt_to_ml/Projects/Lactic_acid/blank/") #select correct directory / todo : change for pipeline
cwd = os.getcwd() + "/"
os.mkdir(cwd + "png/")

#input metadata
parent_dir = os.pardir
input_standards = parent_dir + "/200929_heranalyse_trp_lactic_acid_TM.txt" #metadata has to be in parent directory
df = pd.read_csv(input_standards, sep='\t')

#select correct columns 
df_cols = ["ID", "Metabolite", "Ionisation mode", "Emperical m/z (Da)", "Retention time (min)"]
df = df.loc[:, df_cols]

#rename the columns (for compatibility with existing R-scripts)
df = df.rename(columns={"Metabolite": "MetaboliteName", "Ionisation mode": "IonisationMode", "Emperical m/z (Da)" : "MZ", "Retention time (min)" : "Time" })
#nog wegschrijven 
#df = df.T

#iterate over metabolites metadata
for index, row in df.iterrows():
    # print(row)
    # print("-"*80)
    id = row[0] #save id of metabolite
    metabolite_name = row[1] #save name of metabolite
    ionisation_mode = row[2] #save ionisation mode of metabolite
    MZ = row[3] #save m/z of metabolite

    #remove unneccesary characters
    metabolite_name = re.sub('[\W_]+', '', metabolite_name)
    print(metabolite_name)

    # #change directory
    # os.chdir("/media/sf_SF/Stage2021/Python_code_thermo_txt_to_ml/Testfiles/")

    #loop over the correct df_*.txt files in wd
    directory_list = os.listdir(os.getcwd())
    for filename in directory_list:
        if filename.startswith("df_") and filename.endswith(".txt"):
            print(filename)

            #input dataframe 
            file = cwd + filename
            df_file = pd.read_csv(file, sep='\t')

            #search scannumbers and only retain 1 ionisation mode (from metadata)
            ##positive mode = odd -> 1, negative = even -> 0
            if ionisation_mode == '-':
                df_file = df_file[df_file["scannumber"] % 2 == 0]
            if ionisation_mode == '+':
                df_file = df_file[df_file["scannumber"] % 2 == 1]
                
            ### make block : mz keep +- 0.2 Da arround center
            standard_down = MZ - 0.02 
            standard_up = MZ + 0.02

            #
            roi = df_file[(df_file["m_over_z"] >= standard_down) & (df_file["m_over_z"] <= standard_up)]
            roi = roi.round({'m_over_z': 4, 'time': 2})

            #order from low to high rt
            roi = roi.sort_values('time')

            # #print new df to csv (necessary?)
            # roi.to_csv("roi_"+ metabolite_name + "_" + filename, index=None, sep='\t') 

            # ##create png's
            # #plot
            # plt.figure()
            # plt.scatter(
            #     x=roi.time, 
            #     y=roi.m_over_z,
            #     marker='.',
            #     c=roi.intensity, 
            #     alpha=0.5
            # )
            # plt.colorbar()
            # plt.ylabel("MZ (Da)")
            # plt.xlabel("Time (min)")
            # plt.savefig(cwd + "png/"+ filename + "_" + id + "_" + metabolite_name + '_1.png')

            # #plot + log
            # plt.figure()
            # plt.scatter(
            #     x=roi.time, 
            #     y=roi.m_over_z,
            #     marker='.',
            #     c=np.log10(roi.intensity+1), 
            #     alpha=0.5,
            #     vmin=0, 
            #     vmax=12,
            #     cmap=plt.cm.Blues
            # )

            # plt.colorbar()
            # plt.ylabel("MZ (Da)")
            # plt.xlabel("Time (min)")
            # plt.savefig(cwd + "png/"+ filename + "_" + id + "_" + metabolite_name + '_2.png')

            #plot + log without axis (for ML)
            plt.figure(
                figsize=(7, 5)
            )
            plt.scatter(
                x=roi.time, 
                y=roi.m_over_z,
                marker='.',
                c=np.log10(roi.intensity+1), 
                alpha=0.5,
                vmin=0, 
                vmax=12,
                cmap=plt.cm.Blues
            )
            plt.axis('off')
            plt.savefig(cwd + "png/"+ filename + "_" + id + "_" + metabolite_name + '_3.png')
            plt.close('all')
#close files
