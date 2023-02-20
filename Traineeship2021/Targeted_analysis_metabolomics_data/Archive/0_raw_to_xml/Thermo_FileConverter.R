# File converter 
###################################
# information about script (same as example?)


#########adjustments############

#options
PATH <- 'C:/Users/Loes/Documents/Lactic_acid/' #location of the test folder with the .raw files
PATH_CONVERTER <- 'C:/Users/Loes/Documents/Loes/raw-txt/FileConverter.exe'#location of the exe


##########File_conversion##########
#time
print(Sys.time())
start_time <- Sys.time()
print("R pipeline - Part 0: File_conversion - start!")

#list all directories
print(PATH)
setwd(PATH)
filenames <- list.files(PATH, pattern="*.raw", recursive = F, full.names = T) #not in subfolders; give full path
print(filenames)
#convert all files in dir to .txt
for(file_ in filenames){
  new_filename <- paste0(substr(file_, 1 ,(nchar(file_)-4)), ".txt")
  print(new_filename)
 
  command <- paste0(PATH_CONVERTER, " ", file_, " > ", new_filename)
  print(command)
  shell(command)
}

#time2
print("R pipeline - Part 0: File_conversion - done!")
print(Sys.time())
end_time <- Sys.time()
print(end_time - start_time)

#####################
