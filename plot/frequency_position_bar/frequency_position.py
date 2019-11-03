import numpy as np
import pandas as pd
import sys
import os
path=""
inputname = sys.argv[1]
outputname=sys.argv[2]
sequence=open(inputname)
seq=[]
for line in sequence:
    if line.startswith(">"):
        continue
    else:
        #seq.append(line.replace('\n','').replace('\r',''))
        seq.append(line.replace('\n', '').replace('\r', ''))
numbers = [0]*len(seq[0])
numbres_matrix = []
chars=['A','C','T','G']
frequency_matrix=[[]]*len(chars)
for char_index in range(len(chars)):
    for line in seq:
        for index in range(len(line)):
            if chars[char_index] == line[index]:
                numbers[index] += 1
    numbres_matrix.append(numbers)
    numbers = [0] * len(line)
numbres_matrix = np.array(numbres_matrix)
for row in range(len(numbres_matrix)):
    for colum  in range(len(numbres_matrix[0])):
        frequency_matrix[row]=([e/float(sum(numbres_matrix[:,colum])) for e in numbres_matrix[row]])
frequency_matrix=np.array(frequency_matrix)
print frequency_matrix.shape
pd.DataFrame(frequency_matrix).to_csv(path+outputname,header=None,index=None)
