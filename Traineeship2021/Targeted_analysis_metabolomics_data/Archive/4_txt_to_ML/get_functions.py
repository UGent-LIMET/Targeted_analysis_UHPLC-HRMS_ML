#packages
import os

##function list directories
def directories(path):
    for directory in os.listdir(path):
        if os.path.isdir(os.path.join(path, directory)):
            yield directory

##function list files
def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

