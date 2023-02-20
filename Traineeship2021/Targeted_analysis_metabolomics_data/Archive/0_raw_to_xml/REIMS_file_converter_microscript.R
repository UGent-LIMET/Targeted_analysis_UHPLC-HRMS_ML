# REIMS file converter microscript
###################################
# Title: Software framework for the processing and statistical analysis of multivariate MS-data
# Owner: Laboratory of Chemical Analysis (LCA), Ghent university
# Creator: Dr. Marilyn De Graeve
# Running title: R pipeline
# Script: REIMS file converter microscript

# this microscript runs outside the R pipeline 
# requirments: Windows OS and waters folder present on computer, no spaces in filenames
# to run, open in Rstudio, adjust the 'adjustments' below if needed and click "Source" to run the whole script
# no progressbars are shown, as long as red running symbol is visible, the script is calculating
# check file conversion succesful after finished (no symbols present)



#########adjustments############
#options
PATH_HOST <- 'C:/Users/madgraev/Desktop' # pc_groot, persoonlijke pc van mdg
PATH_DI06C001 <- 'C:/Users/Marilyn/Documents/Files/Werk_LCA/Pipeline_metabolomics/Data/Input/20210326_IC_specificationfish/bio'
#'C:/Users/Marilyn/Desktop/REIMS_file_conversion_microscript/test4'
#'C:/Users/Marilyn/Documents/Files/Werk_LCA/Pipeline_metabolomics/Data/Input/20210308_VG_30SummerBreedSamples/bio'
#"C:/Users/Marilyn/Desktop/Projects/2020/202003.PRO/Data"
#'C:/Users/Marilyn/Desktop/REIMS_file_conversion_microscript/test' # professional pc van lca voor mdg
PATH_RBOX02 <- '/home/lca/Documents' # Rbox2 16.04 LTS
PATH_OPERA <- 'C:/Users/UGent/Documents/Pipeline_metabolomics/Data/Input/20210326_IC_specificationfish/bio' #opera laptop van lca labo
PATH_REIMS_PC_RIGHT <- 'C:/Users/Administrator/Desktop/REIMS_file_conversion_microscript/test'
PATH_HDRIVE <- "D:/ToestelData/PCGC24/2020/Projects/2020/202001.PRO/Data"

#adjustments
PATH <- PATH_DI06C001 #location of folder with the .raw files

PATH_WATERSCONVERTER <- "C:/Users/Marilyn/Desktop/REIMS_file_conversion_microscript/waters/WatersStringDump.exe"
#"C:/Users/Marilyn/Desktop/REIMS_file_conversion_microscript/waters/WatersStringDumpWithDaughter.exe"  #mrm files, ms2 w transitions
#"C:/Users/Marilyn/Desktop/REIMS_file_conversion_microscript/waters/WatersStringDump_with_debug.exe" #location convertion program, not in use
#"C:/Users/Marilyn/Desktop/REIMS_file_conversion_microscript/waters/WatersStringDump.exe"   #default progr

#
#####################




# => DO NOT ADJUST CODE BELOW THIS POINT!
##########File_conversion##########
#time
print(Sys.time())
start_time <- Sys.time()
print("R pipeline - Part 0: File_conversion - start!")

#list all directories
setwd(PATH)
filenames <- list.files(PATH, pattern="*.raw", recursive = F, full.names = T) #not in sobfolders; give full path

#convert all files in dir to .txt
for(file_ in filenames){
  new_filename <- paste0(substr(file_, 1 ,(nchar(file_)-4)), ".txt")
  #command:
  #C:\Users\Marilyn\Desktop\waters\WatersStringDump.exe C:\Users\Marilyn\Desktop\200323_A1_102_neg_001.raw > C:\Users\Marilyn\Desktop\200323_A1_102_neg_001.txt
  command <- paste0(PATH_WATERSCONVERTER, " ", file_, " > ", new_filename)
  shell(command)
}

#time2
print("R pipeline - Part 0: File_conversion - done!")
print(Sys.time())
end_time <- Sys.time()
print(end_time - start_time)

#
#####################
