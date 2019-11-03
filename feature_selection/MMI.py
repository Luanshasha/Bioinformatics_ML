import numpy as np
import pandas as pd
import sys
import math
import itertools
path=""
inputname=sys.argv[1]
outputname=sys.argv[2]
sequence=open(path+inputname)
seq=[]
for line in sequence:
    if line.startswith(">"):
        continue
    else:
        seq.append(line.replace('\r','').replace('\n',''))

chars = ['A', 'C', 'G', 'T']
one_char=[''.join(x) for x in itertools.combinations_with_replacement(chars, 1)]
two_chars = [''.join(x) for x in itertools.combinations_with_replacement(chars, 2)]
three_chars = [''.join(x) for x in itertools.combinations_with_replacement(chars, 3)]
results=[]
for line in seq:
    one = [0] * len(one_char)
    two_combin = [0] * len(two_chars)
    three_combin = [0] * len(three_chars)
    frequency_one_nolamdon=[0]*len(one_char)
    frequency_one = [0] * len(one_char)
    frequency_two_combin = [0] * len(three_chars)
    three_combin = [0] * len(three_chars)
    I_two = [0] * len(two_chars)
    I_three = [0] * len(three_chars)
    I_whole = []
    lamda=0.001
    for index in range(len(line)):
        if line[index] in one_char:
            one[one_char.index(line[index])] += 1
        if line[index:index+2] in two_chars :
            two_combin[two_chars.index(line[index:index+2])] += 1
        if line[index:index+2] not in two_chars and line[index:index+2][::-1] in two_chars :
            two_combin[two_chars.index(line[index:index + 2][::-1])] += 1
        if line[index:index+3]  in three_chars:
            three_combin[three_chars.index(line[index:index+3])] +=1
        if line[index:index+3]  not in three_chars and index<len(line)-2:
            list=[''.join(x) for x in itertools.permutations(line[index:index+3])]
            for e in list:
                if e in three_chars:
                    three_combin[three_chars.index(e)] +=1
                    break
    frequency_one_nolamdon = [e  / float(sum(one)) for e in one]
    frequency_one = [(e+lamda)/float(sum(one)+lamda) for e in one]
    frequency_two_combin = [(e+lamda) / float(sum(two_combin)+lamda) for e in two_combin]
    frequency_three_combin = [(e+lamda) / float(sum(three_combin)+lamda) for e in three_combin]
    for two_chars_index in range(len(two_chars)):
        I_two[two_chars_index] = frequency_two_combin[two_chars_index]*np.log(frequency_two_combin[two_chars_index]/(frequency_one[one_char.index(two_chars[two_chars_index][0])]*frequency_one[one_char.index(two_chars[two_chars_index][1])]))
    for three_chars_index in range(len(three_chars)):
        first_expression = frequency_two_combin[two_chars.index(three_chars[three_chars_index][:2])]*math.log(frequency_two_combin[two_chars.index(three_chars[three_chars_index][:2])]/(frequency_one[one_char.index(three_chars[three_chars_index][0])]*frequency_one[one_char.index(three_chars[three_chars_index][1])]))
        second_expression = (frequency_two_combin[two_chars.index(three_chars[three_chars_index][0]+three_chars[three_chars_index][2])]/frequency_one[one_char.index(three_chars[three_chars_index][2])])*math.log(frequency_two_combin[two_chars.index(three_chars[three_chars_index][0]+three_chars[three_chars_index][2])]/frequency_one[one_char.index(three_chars[three_chars_index][2])])
        third_expression = (frequency_three_combin[three_chars_index]/frequency_two_combin[two_chars.index(three_chars[three_chars_index][1:])])*math.log(frequency_three_combin[three_chars_index]/frequency_two_combin[two_chars.index(three_chars[three_chars_index][1:])])
        #I_three[three_chars_index] = (frequency_two_combin[two_chars.index(three_chars[three_chars_index][:2])]*math.log(frequency_two_combin[two_chars.index(three_chars[three_chars_index][:2])]/(frequency_one[one_char.index(three_chars[three_chars_index][0])]*frequency_one[one_char.index(three_chars[three_chars_index][1])]))+
        #(frequency_two_combin[two_chars.index(three_chars[three_chars_index][0]+three_chars[three_chars_index][2])]/frequency_one[one_char.index(three_chars[three_chars_index][2])])*math.log(frequency_two_combin[two_chars.index(three_chars[three_chars_index][0]+three_chars[three_chars_index][2])]/frequency_one[one_char.index(three_chars[three_chars_index][2])])-
        #(frequency_three_combin[three_chars_index]/frequency_two_combin[two_chars.index(three_chars[three_chars_index][1:])])*math.log(frequency_three_combin[three_chars_index]/frequency_two_combin[two_chars.index(three_chars[three_chars_index][1:])]))
        I_three[three_chars_index] = first_expression + second_expression - third_expression
    I_whole.extend(I_three)
    I_whole.extend(I_two)
    I_whole.extend(frequency_one_nolamdon)
    results.append([round(e,8) for e in I_whole])
print(np.array(results).shape)
pd.DataFrame(results).to_csv(outputname, header=None, index=None)
sequence.close()







