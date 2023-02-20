# Title: Software framework for the processing and statistical analysis of multivariate MS-data
# Owner: Laboratory of Chemical Analysis (LCA), Ghent university
# Creator: Dr. Marilyn De Graeve
# Running title: R pipeline
# Script: Part 0: File_conversion


##########R Pipeline - Part 0: File_conversion##########
print(Sys.time())
start_time <- Sys.time()
print("R pipeline - Part 0: File_conversion - start!")
# Part I: File_conversion


## data_loading
setwd(path_data_in)

# convert raw files and raw directories:
if(INSTRUMENT == EXACTIVE){
 # converts raw file -> mzXML file
  
  # unzip files if needed
  filenames <- list.files(path_data_in, pattern="*.raw", recursive = TRUE) #exactive stalen
  path_data_in_bio <- file.path(path_data_in, 'bio.zip') 
  path_data_in_blank <- file.path(path_data_in, 'blank.zip')
  if(length(filenames) == 0){
    unzip(path_data_in_bio, exdir=path_data_in)
    unzip(path_data_in_blank, exdir=path_data_in)
    filenames <- list.files(path_data_in, pattern="*.raw", recursive = TRUE)
  }

  ## set path to find files
  path_data_in_bio <- file.path(path_data_in, 'bio') 
  path_data_in_blank <- file.path(path_data_in, 'blank')
  setwd(path_data_in_bio)
  
  ## convert files in corresponding folder
  #http://proteowizard.sourceforge.net/tools/filters.html
  if (CODE_RUN_MODE == CODE_DEVELOPMENT){ 
    #via windows install on PC, ev. adjust path to software msconvert.exe
    setwd(path_data_in_bio)
    if(PEAK_PICKING_MODE == CENTROID){
      tryCatch({shell(paste0('"', '"', PATH_MSCONVERT, '" ', '"', '" --mzXML --64 --filter "peakPicking centroid msLevel=1-" --filter "polarity "', POLARITY ,'"', '" *.RAW -o "', path_data_in_bio,'"'))})
    }
    if(PEAK_PICKING_MODE == PROFILE){
      tryCatch({shell(paste0('"', '"', PATH_MSCONVERT, '" ', '"', '" --mzXML --64 --filter "peakPicking cwt snr=0.1 peakSpace=0.1 msLevel=1-" --filter "polarity "', POLARITY ,'"', '" *.RAW -o "', path_data_in_bio,'"'))})
    }
    
    setwd(path_data_in_blank)
    if(PEAK_PICKING_MODE == CENTROID){
      tryCatch({shell(paste0('"', '"', PATH_MSCONVERT, '" ', '"', '" --mzXML --64 --filter "peakPicking centroid msLevel=1-" --filter "polarity "', POLARITY ,'"', '" *.RAW -o "', path_data_in_blank,'"'))})
    }
    if(PEAK_PICKING_MODE == PROFILE){
      tryCatch({shell(paste0('"', '"', PATH_MSCONVERT, '" ', '"', '" --mzXML --64 --filter "peakPicking cwt snr=0.1 peakSpace=0.1 msLevel=1-" --filter "polarity "', POLARITY ,'"', '" *.RAW -o "', path_data_in_blank,'"'))})
    }
  }
  if (CODE_RUN_MODE == CODE_AUTORUN){
    # via linux terminal, no install but pull msconvert code from docker
    # sudo docker run -it --rm -e WINEDEBUG=-all -v /your/data:/data chambm/pwiz-skyline-i-agree-to-the-vendor-licenses wine msconvert /data/file.raw
    # https://hub.docker.com/r/chambm/pwiz-skyline-i-agree-to-the-vendor-licenses
    setwd(path_data_in_bio)
    if(PEAK_PICKING_MODE == CENTROID){
      print("STOP! MSconvert does not (yet) work for linux")
      stop("STOP! does not work for linux")
      #tryCatch({system(paste0('sudo docker run -t --rm -e WINEDEBUG=-all -v ', 
      #                       path_data_in_bio, ' chambm/pwiz-skyline-i-agree-to-the-vendor-licenses wine msconvert', 
      #                       ' --mzXML --64 --filter "peakPicking centroid msLevel=1-" --filter "polarity ', POLARITY ,'" *.RAW -o ', path_data_in_bio))})
    }
    if(PEAK_PICKING_MODE == PROFILE){
      tryCatch({system(paste0('"', '"', PATH_MSCONVERT, '" ', '"', '" --mzXML --64 --filter "peakPicking cwt snr=0.1 peakSpace=0.1 msLevel=1-" --filter "polarity "', POLARITY ,'"', '" *.RAW -o "', path_data_in_bio,'"'))})
    }
    
    setwd(path_data_in_blank)
    if(PEAK_PICKING_MODE == CENTROID){
      tryCatch({system(paste0('"', '"', '"sudo docker run -it --rm -e WINEDEBUG=-all -v "',
                             path_data_in_bio, ' "chambm/pwiz-skyline-i-agree-to-the-vendor-licenses wine msconvert"',
                             '" ', '"', '" --mzXML --64 --filter "peakPicking centroid msLevel=1-" --filter "polarity "', POLARITY ,'"', '" *.RAW -o "', path_data_in_blank,'"'))})
    }
    if(PEAK_PICKING_MODE == PROFILE){
      tryCatch({system(paste0('"', '"', PATH_MSCONVERT, '" ', '"', '" --mzXML --64 --filter "peakPicking cwt snr=0.1 peakSpace=0.1 msLevel=1-" --filter "polarity "', POLARITY ,'"', '" *.RAW -o "', path_data_in_blank,'"'))})
    }
  }
}



if(INSTRUMENT == REIMS){
  # converts raw directory -> txt file 

  #todo add stopifnot(): check NO spaces in filenames!
  
  #does not work in Linux, since depends on win dependencies (exe, dll, lib, ...)
  if (CODE_RUN_MODE == CODE_AUTORUN){
    print("STOP! REIMS file converter does not work in linux")
    stop("STOP! REIMS file converter does not work for linux")
  }


  #list all directories
  path_data_in_bio <- file.path(path_data_in, 'bio')
  setwd(path_data_in_bio)
  filenames <- list.files(path_data_in_bio, pattern="*.raw", recursive = F, full.names = T) #not in sobfolders; give full path
  
  #convert all files in dir to .txt
  for(file_ in filenames){
    new_filename <- paste0(substr(file_, 1 ,(nchar(file_)-4)), ".txt")
    #command:
    #C:\Users\Marilyn\Desktop\waters\WatersStringDump.exe C:\Users\Marilyn\Desktop\200323_A1_102_neg_001.raw > C:\Users\Marilyn\Desktop\200323_A1_102_neg_001.txt
    command <- paste0(PATH_WATERSCONVERTER, " ", file_, " > ", new_filename)
    shell(command)
  }
}

#todo: add more vendors in future (also diff for waters with rt = see archive using msconvert as well).



print("R pipeline - Part 0: File_conversion - done!")
print(Sys.time())
end_time <- Sys.time()
print(end_time - start_time)
#
#####################
