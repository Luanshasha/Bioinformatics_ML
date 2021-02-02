import getopt
import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import MinMaxScaler
whole_result=[]
input_files=""
whole_dimension=[]

def feature_combine():
    nowpath = os.getcwd()+ "\\" + 'feature_extraction/'
    input_files = os.listdir(nowpath)
    print(os.getcwd())
    output_name ="combine.csv"
    for i in range(len(input_files)):
        for j in range(i+1,len(input_files)):
            print(input_files)
            first_file = pd.read_csv(nowpath+input_files[i], header=None, index_col=None)  # sys.argv[1])
            first_file = first_file.values[:, 0:]
            first_file = pd.DataFrame(first_file).astype(float)

            second_file = pd.read_csv(nowpath+input_files[j], header=None, index_col=None)
            second_file=second_file.values[:,0:]
            second_file = pd.DataFrame(second_file).astype(float)

            print("first_file_num:", len(first_file))
            print ("first_file_length:", len(first_file.values[0]))
            print ("second_file_num:", len(second_file))
            print ("second_file_length:", len(second_file.values[0]))
            output_file = pd.concat([first_file, second_file], axis=1)
            print ("output_file_num:", len(output_file))
            print ("output_file_len:", len(output_file.values[0]))
            scaler = MinMaxScaler()
            output_file = scaler.fit_transform(np.array(output_file))
            print ("normalization")
            print(output_file.shape)
            pd.DataFrame(output_file).to_csv(os.getcwd()+ "\\" +output_name, header=None, index=False)