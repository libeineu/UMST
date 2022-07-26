import os
import numpy as np
import codecs
root_path = 'path/to/raw/bpe-file'
dirs = os.listdir(root_path)
print(dirs)
for filename in dirs:
    file_path = root_path + "/" + filename
    f = open(file_path, "r")
    output_path = root_path + "/" + filename + "." + "tree"
    output = codecs.open(output_path, 'w', encoding='utf-8')
    for line in f.readlines():
        line = line.strip("\n")
        line = line.split(" ")
        mapping = ""
        idx = 1
        for word in line:
            if word[-2:] == "@@":
               mapping+=(str(idx))
               mapping += " "
               continue
            else:
               mapping += str(idx)
               mapping += " "
               idx += 1
               continue
        mapping = mapping[:-1]
        mapping += "\n"
        output.write(mapping)