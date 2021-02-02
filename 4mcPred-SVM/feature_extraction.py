# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import itertools
import os

def feature_extract(input_data,gene_type,fill_NA):
    nowpath = os.getcwd()
    features_extraction = nowpath + "\\" + 'feature_extraction/'
    os.mkdir(features_extraction)

    ###============================AthMethPre=================================
    print("AthMethPre")
    dataset_name = "trainset"
    output_name= input_data.split(".")[0]+"_AthMethPre.csv"
    type_value = "U"
    if gene_type == "RNA":
        type_value = "U"
    elif gene_type == "DNA":
        type_value = "T"

    def read_fasta_file(path):
        fh = open(m6a_benchmark_dataset)
        seq = []
        for line in fh:
            if line.startswith('>'):
                continue
            else:
                seq.append(line.replace('\n', '').replace('\r', ''))
        fh.close()
        #matrix_data = np.array([list(e) for e in seq])
        matrix_data = []
        for e in seq:
            temp = []
            for i in e:
                temp.append(i)
            matrix_data.append(temp)
        return matrix_data

    def AthMethPre_extract_one_line(data_line):
        A = [0, 0, 0, 1]
        T = [0, 0, 1, 0]
        C = [0, 1, 0, 0]
        G = [1, 0, 0, 0]
        N = [0, 0, 0, 0]
        feature_representation = {"A": A, "C": C, "G": G, "N": N}
        feature_representation[type_value] = T
        beginning = 0
        end = len(data_line) - 1
        one_line_feature = []
        alphabet = 'ACNG'
        alphabet += type_value
        matrix_two = ["".join(e) for e in itertools.product(alphabet, repeat=2)]  # AA AU AC AG UU UC ...
        matrix_three = ["".join(e) for e in itertools.product(alphabet, repeat=3)]  # AAA AAU AAC ...
        matrix_four = ["".join(e) for e in itertools.product(alphabet, repeat=4)]  # AAAA AAAU AAAC ...
        feature_two = np.zeros(25)
        feature_three = np.zeros(125)
        feature_four = np.zeros(625)
        for index, data in enumerate(data_line):
            if index == beginning or index == end:
                one_line_feature.extend(feature_representation["N"])
            elif data in feature_representation.keys():
                one_line_feature.extend(feature_representation[data])
            if "".join(data_line[index:(index + 2)]) in matrix_two and index <= end - 1:
                feature_two[matrix_two.index("".join(data_line[index:(index + 2)]))] += 1
            if "".join(data_line[index:(index + 3)]) in matrix_three and index <= end - 2:
                feature_three[matrix_three.index("".join(data_line[index:(index + 3)]))] += 1
            if "".join(data_line[index:(index + 4)]) in matrix_four and index <= end - 3:
                feature_four[matrix_four.index("".join(data_line[index:(index + 4)]))] += 1
        sum_two = np.sum(feature_two)
        sum_three = np.sum(feature_three)
        sum_four = np.sum(feature_four)
        one_line_feature.extend(feature_two / sum_two)
        one_line_feature.extend(feature_three / sum_three)
        one_line_feature.extend(feature_four / sum_four)
        return one_line_feature


    def AthMethPre_extract_one_line_without(data_line):
        A = [0, 0, 0, 1]
        U = [0, 0, 1, 0]
        C = [0, 1, 0, 0]
        G = [1, 0, 0, 0]
        N = [0, 0, 0, 0]
        feature_representation = {"A": A, "C": C, "G": G, "N": N}
        feature_representation[type_value] = U
        beginning = 0
        end = len(data_line) - 1
        one_line_feature = []
        alphabet = 'ACG'
        alphabet += type_value
        matrix_two = ["".join(e) for e in itertools.product(alphabet, repeat=2)]  # AA AU AC AG UU UC ...
        matrix_three = ["".join(e) for e in itertools.product(alphabet, repeat=3)]  # AAA AAU AAC ...
        matrix_four = ["".join(e) for e in itertools.product(alphabet, repeat=4)]  # AAAA AAAU AAAC ...
        feature_two = np.zeros(16)
        feature_three = np.zeros(64)
        feature_four = np.zeros(256)
        for index, data in enumerate(data_line):
            if data in feature_representation.keys():
                one_line_feature.extend(feature_representation[data])
            if "".join(data_line[index:(index + 2)]) in matrix_two and index <= end - 1:
                feature_two[matrix_two.index("".join(data_line[index:(index + 2)]))] += 1
            if "".join(data_line[index:(index + 3)]) in matrix_three and index <= end - 2:
                feature_three[matrix_three.index("".join(data_line[index:(index + 3)]))] += 1
            if "".join(data_line[index:(index + 4)]) in matrix_four and index <= end - 3:
                feature_four[matrix_four.index("".join(data_line[index:(index + 4)]))] += 1
        sum_two = np.sum(feature_two)
        sum_three = np.sum(feature_three)
        sum_four = np.sum(feature_four)
        one_line_feature.extend(feature_two / sum_two)
        one_line_feature.extend(feature_three / sum_three)
        one_line_feature.extend(feature_four / sum_four)
        return one_line_feature

    def AthMethPre_feature_extraction(matrix_data, fill_NA):
        if fill_NA == "1":
            final_feature_matrix = [AthMethPre_extract_one_line(e) for e in matrix_data]
        elif fill_NA == "0":
            final_feature_matrix = [AthMethPre_extract_one_line_without(e) for e in matrix_data]
        return final_feature_matrix

    if fill_NA == "1":
        m6a_benchmark_dataset = input_data
        matrix_data = read_fasta_file(m6a_benchmark_dataset)
        AthMethPre_feature_matrix = AthMethPre_feature_extraction(matrix_data, fill_NA)
        print(np.array(AthMethPre_feature_matrix).shape)
        pd.DataFrame(AthMethPre_feature_matrix).to_csv(features_extraction+output_name, header=None, index=False)
    elif fill_NA == "0":
        m6a_benchmark_dataset = input_data
        matrix_data = read_fasta_file(m6a_benchmark_dataset)
        AthMethPre_feature_matrix = AthMethPre_feature_extraction(matrix_data, fill_NA)
        print(np.array(AthMethPre_feature_matrix).shape)
        pd.DataFrame(AthMethPre_feature_matrix).to_csv(features_extraction+output_name, header=None, index=False)

    ###============================2RFH=================================
    print("2RFH")
    path=""
    mark_n = False
    output_name= input_data.split(".")[0]+"_"+"2RFH.csv"

    def read_fasta_file(path):
        global mark_n
        fh = open(path)
        seq = []
        for line in fh:
            if line.startswith('>'):
                continue
            else:
                temp_data = ["U" if e == "T" else e for e in line.replace('\r', '').replace('\n', '')]
                if "N" in temp_data:
                    mark_n = True
                seq.append(temp_data)

        fh.close()
        #matrix_data = np.array([list(e) for e in seq])
        matrix_data = []
        for e in seq:
            temp = []
            for i in e:
                temp.append(i)
            matrix_data.append(temp)
        return matrix_data


    def fetch_singleline_features_withN(sequence):
        alphabet = "AUGCN"
        k_num = 2
        char_num = 5
        two_sequence = []
        for index, data in enumerate(sequence):
            if index < (len(sequence) - k_num + 1):
                two_sequence.append("".join(sequence[index:(index + k_num)]))
        parameter = [e for e in itertools.product([0, 1], repeat=char_num)][:int(pow(char_num, k_num))]
        record = [0 for x in range(int(pow(char_num, k_num)))]
        matrix = ["".join(e) for e in itertools.product(alphabet, repeat=k_num)]  # AA AU AC AG UU UC ...
        final = []
        for index, data in enumerate(two_sequence):
            if data in matrix:
                final.extend(parameter[matrix.index(data)])
                record[matrix.index(data)] += 1
                final.append(record[matrix.index(data)] * 1.0 / (index + 1))
        return final


    def fetch_singleline_features_withoutN(sequence):
        alphabet = "AUGC"
        k_num = 2
        two_sequence = []
        for index, data in enumerate(sequence):
            if index < (len(sequence) - k_num + 1):
                two_sequence.append("".join(sequence[index:(index + k_num)]))
        parameter = [e for e in itertools.product([0, 1], repeat=4)]
        record = [0 for x in range(int(pow(4, k_num)))]
        matrix = ["".join(e) for e in itertools.product(alphabet, repeat=k_num)]  # AA AU AC AG UU UC ...
        final = []
        for index, data in enumerate(two_sequence):
            if data in matrix:
                final.extend(parameter[matrix.index(data)])
                record[matrix.index(data)] += 1
                final.append(record[matrix.index(data)] * 1.0 / (index + 1))
        return final

    sequences = read_fasta_file(path + input_data)
    two_RFH_features_data = []
    if mark_n == True:
        for index, sequence in enumerate(sequences):
            two_RFH_features_data.append(fetch_singleline_features_withN(sequence))
    else:
        for index, sequence in enumerate(sequences):
            two_RFH_features_data.append(fetch_singleline_features_withoutN(sequence))
    print(np.array(two_RFH_features_data).shape)
    print("mark_n", mark_n)
    pd.DataFrame(two_RFH_features_data).to_csv(features_extraction + output_name, header=None, index=None)

