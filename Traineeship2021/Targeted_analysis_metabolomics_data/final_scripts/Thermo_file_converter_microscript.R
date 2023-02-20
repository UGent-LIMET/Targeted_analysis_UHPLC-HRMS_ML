# Title: Software framework for the processing and statistical analysis of multivariate MS-data
# Owner: Laboratory of Chemical Analysis (LCA), Ghent University
# Creator: Dr. Marilyn De Graeve, Loes Vervaele
# Running title: R pipeline
# Script: Part 0: Thermo file conversion using MSConvert (ProteoWizard)

# This script runs outside the R pipeline.
# Requirements: Windows OS and Thermo folder present on computer, no spaces in filenames, msconvert.exe installed
# To run, open in Rstudio, adjust the 'adjustments' below if needed and click "Source" to run the whole script
# No progressbars are shown, as long as red running symbol is visible, the script is calculating
# Check file conversion successful after finished (no symbols present)



########## Adjustments ##########

PATH_MSCONVERT <- 'C:/Windows/System32/MSConvert/msconvert.exe' # path to msconvert.exe
INPUT <- 'C:/Users/Loes/OneDrive - Hogeschool West-Vlaanderen/Vervaele Loes/0. Dashboard project/0. Testfiles/200929s002/' # path to .raw files
OUTPUT <- 'C:/Users/Loes/OneDrive - Hogeschool West-Vlaanderen/Vervaele Loes/0. Dashboard project/0. Testfiles/200929s002/' # path to the output folder
POLARITY <- 'negative'


#choose setting
OPTION1 <- 'pre-processing using R pipeline'
OPTION2 <- 'targeted analysis using ML'
setting <- OPTION2 # choose the wanted setting (OPTION1 or OPTION2)


# if OPTION1 : choose peak picking mode 
CENTROID <- 'peak picking in centroid mode'
PROFILE <- 'peak picking in profile mode'
PEAK_PICKING_MODE <- CENTROID 


########## 




# => DO NOT ADJUST CODE BELOW THIS POINT!
##########R Pipeline - Part 0: File_conversion##########
print(Sys.time())
start_time <- Sys.time()
print("R pipeline - Part 0: File_conversion - start!")
# Part I: File_conversion


#settings fixed
path_data_in <- INPUT
path_data_out <- OUTPUT


## setting working directory
setwd(path_data_in)


## converts raw file -> mzXML file
# selecting files 
filenames <- list.files(path_data_in, pattern="*.raw", recursive = TRUE) #exactive stalen
print(filenames)

# loop over the files
# selecting the correct settings 
for (x in filenames){
  if(setting == OPTION1){
    if(PEAK_PICKING_MODE == CENTROID){
      tryCatch({shell(paste0('"', '"', PATH_MSCONVERT, '" ', '"', '" --mzXML --64 --filter "peakPicking centroid msLevel=1-" --filter "polarity "', POLARITY ,'"', '" *.RAW -o "', path_data_out,'"'))})
    }
    if(PEAK_PICKING_MODE == PROFILE){
      tryCatch({shell(paste0('"', '"', PATH_MSCONVERT, '" ', '"', '" --mzXML --64 --filter "peakPicking cwt snr=0.1 peakSpace=0.1 msLevel=1-" --filter "polarity "', POLARITY ,'"', '" *.RAW -o "', path_data_out,'"'))})
    }
  }
  if(setting == OPTION2){
    tryCatch({shell(paste0('"', '"', PATH_MSCONVERT, '" ', '"', '" --mzXML --64 "', '" *.RAW -o "', path_data_out,'"'))})
  }
}

print("R pipeline - Part 0: File_conversion - done!")
print(Sys.time())
end_time <- Sys.time()
print(end_time - start_time)

#####################
