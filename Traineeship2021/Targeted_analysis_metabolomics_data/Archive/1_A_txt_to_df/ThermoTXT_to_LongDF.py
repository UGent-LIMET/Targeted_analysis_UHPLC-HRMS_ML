#!/usr/bin/python3

#packages
import pandas as pd

##function txt to df
def ThermoTXT_to_LongDF(filename):
    df_cols = ["scannumber", "time", "intensity", "m_over_z"] #make new df with correct column names
    global df
    df = []
    current_scan = 0
    open_file = open(filename,"r")
    for line in open_file: #read all the lines of the txt files

        #append scan no, start time, intensity and m_over_z to df
        if "ScanHeader" in line:
            current_scan = int(line.split("#")[1].strip())
            print(f"{current_scan}")
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
    new_filename = "df_test_" + filename
    df2.to_csv(new_filename, header=df_cols, index=None, sep='\t') 

   