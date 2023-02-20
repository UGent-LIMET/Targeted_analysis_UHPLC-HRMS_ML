#packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as m
import re
import os
from PIL import Image

##function txt to df
def txt_to_df(file, path):
    df_cols = ["scannumber", "time", "intensity", "m_over_z"] #make new df with correct column names
    global df
    df = []
    current_scan = 0
    open_file = open(path+file,"r")

    for line in open_file: #read all the lines of the txt files
        #append scan no, start time, intensity and m_over_z to df
        if "ScanHeader" in line:
            current_scan = int(line.split("#")[1].strip())
            #print(f"{current_scan}")
        if "start_time" in line:
            time = str(line.split(", ")[0].strip())
            time = float(time.split("= ")[1].strip())
        if ' intensity =' in line:
            _, intensity, m_over_z = line.split(",")
            intensity = float(intensity.replace("intensity = ", "").strip())
            m_over_z = float(m_over_z.replace("mass/position = ", "").strip())
            df.append([
                current_scan,
                time,
                intensity,
                m_over_z
            ])
    open_file.close() #close the opened file
    
    #add colnames + make new csv       
    df2 = pd.DataFrame(df, columns=df_cols)
    new_file = path + "df_" + file
    df2.to_csv(new_file, header=df_cols, index=None, sep='\t') 
    
##function df to png
def df_to_png (META, file, path):
    #import metadata 
    df_meta = pd.read_csv(META, sep=',')
    
    #iterate over metadata
    for index, row in df_meta.iterrows():
        met_id = row[1] #save id of metabolite
        metabolite_name = row[2] #save name of metabolite
        ionisation_mode = row[3] #save ionisation mode of metabolite
        MZ = row[4] #save m/z of metabolite
        
        #remove unneccesary characters
        metabolite_name = re.sub('[\W_]+', '', metabolite_name)

        #input dataframe 
        filename = path + file
        df_file = pd.read_csv(filename, sep='\t')

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
        plt.savefig(path + "png/"+ file + "_" + met_id + "_" + metabolite_name + '.png')
        plt.close('all')
    #close files

def resize_images(src_path, dst_path):
    # Here src_path is the location where images are saved.
    for image in os.listdir(src_path):
        if image.endswith(".png"): 
            open_img=Image.open(src_path+image)
            new_img = open_img.resize((64,64)) 
            new_img.save(dst_path+image)

# get data from directory
def get_data(path):
    all_images_as_array=[]
    label=[]
    for filename in os.listdir(path):
        try:
            if re.search(r'FOUND',filename): 
                label.append(1)
            else:
                label.append(0)
            img=Image.open(path + filename)
            np_array = np.asarray(img)
            l,b,c = np_array.shape    
            np_array = np_array.reshape(l*b*c,)   
            all_images_as_array.append(np_array)
        except:
            print(filename) #if error with 2dim, print
            continue

    return np.array(all_images_as_array), np.array(label)

# functie in functie?
# get data from list 
def get_data_rf(path, sel_list):
    all_images_as_array=[]
    label=[]
    for filename in sel_list:
        lm = 2
        try:
            if "FOUND" in filename:
                label.append(1)
                img=Image.open(path + "found/" + filename)
                np_array = np.asarray(img)
                l,b,c = np_array.shape    
                np_array = np_array.reshape(l*b*c,)   
                all_images_as_array.append(np_array)
            else:
                label.append(0)
                img=Image.open(path + "not_found/" + filename)
                np_array = np.asarray(img)
                l,b,c = np_array.shape    
                np_array = np_array.reshape(l*b*c,)   
                all_images_as_array.append(np_array)
        except:
            print("mistake :  " + filename) #if error with 2dim, print
            continue
    return np.array(all_images_as_array), np.array(label)
