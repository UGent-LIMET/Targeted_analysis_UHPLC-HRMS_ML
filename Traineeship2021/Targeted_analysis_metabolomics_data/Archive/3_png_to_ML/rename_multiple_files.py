#!/usr/bin/python3

import os
import re

def rename_multiple_files(path,obj):

    i=0

    for filename in os.listdir(path):
        if re.search("FOUND", filename):
            try:
                f,extension = os.path.splitext(path+filename)
                src=path+filename
                dst=path+filename+obj+str(i)+extension
                os.rename(src,dst)
                i+=1
            except:
                i+=1
