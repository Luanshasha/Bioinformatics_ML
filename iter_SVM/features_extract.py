# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import itertools
import os

def feature_extract(input_data,gene_type,fill_NA):
    nowpath = os.getcwd()
    features_extraction = nowpath + "\\" + 'feature_extraction/'
    os.mkdir(features_extraction)
    ###============================PseEIIP=================================
    print("PseEIIP")
    outputname=input_data.split(".")[0]+"_"+"PseEIIP.csv"
    gene_value="U"
    if gene_type=="RNA":
        gene_value="U"
    elif gene_type=="DNA":
        gene_value="T"

    def read_fasta_file(input_data):
        fh=open(input_data)
        seq=[]
        for line in fh:
            if line.startswith('>'):
                continue
            else:
                seq.append(line.replace('\r','').replace('\n',''))
        fh.close()
        matrix_data = []
        # matrix_data=np.array([list(e) for e in seq])
        for e in seq:
            temp = []
            for i in e:
                temp.append(i)
            matrix_data.append(temp)
        matrix_data = np.array(matrix_data)

        return matrix_data


    def AthMethPre_extract_one_line(data_line):
        end=len(data_line)-1
        one_line_feature=[]
        alphabet='ACG'
        alphabet+=gene_value
        if fill_NA=="1":
            alphabet+="N"
        matrix_three=["".join(e) for e in itertools.product(alphabet, repeat=3)]# AAA AAU AAC ...
        feature_three=np.zeros(len(matrix_three))
        A=0.1260
        U=0.1335
        C=0.1340
        G=0.0806
        N=0.0000
        temp=[A,C,G,U]
        if fill_NA=="1":
            temp.append(N)
        AUCG=[sum(e) for e in itertools.product(temp, repeat=3)]# AAA AAU AAC ...
        for index,data in enumerate(data_line):
            if "".join(data_line[index:(index+3)]) in matrix_three and index <= end-2:
                feature_three[matrix_three.index("".join(data_line[index:(index+3)]))]+=1
        sum_three=np.sum(feature_three)
        feature_three=feature_three/sum_three
        feature_three=feature_three*AUCG
        one_line_feature.extend(feature_three)
        return one_line_feature

    def AthMethPre_feature_extraction(matrix_data):
        final_feature_matrix=[AthMethPre_extract_one_line(e) for e in matrix_data]
        return final_feature_matrix

    matrix_data=read_fasta_file(input_data)
    PseEIIP_features_data=AthMethPre_feature_extraction(matrix_data)
    print(np.array(PseEIIP_features_data).shape)
    pd.DataFrame(PseEIIP_features_data).to_csv(features_extraction+outputname,header=None,index=False)


    ###============================RFH=================================
    print("RFH")
    path = ''
    outputname=input_data.split(".")[0]+"_"+"RFH.csv"
    gene_value = "U"
    if gene_type == "RNA":
        gene_value = "U"
    elif gene_type == "DNA":
        gene_value = "T"

    RFH_features_data = []
    def convert_with(dataPath, outputPath):
        """RFH feature"""
        lines = open(dataPath).readlines()
        finally_text = open(features_extraction+outputPath, 'w')
        finnaly_lines = ""
        for line in lines:
            if line.strip() == "": continue
            if line.strip()[0] in ['A', 'G', 'C', gene_value, 'N']:
                position_mark = 0
                count_AGCT = [0, 0, 0, 0, 0]
                temp = ""
                re = []
                for i in line.strip():
                    re.append(i)
                for x in re:
                #for x in list(line.strip()):
                    position_mark += 1
                    if x == "A" or x == "G":
                        temp += "1,"
                    else:
                        temp += "0,"
                    if x == "A" or x == gene_value:
                        temp += "1,"
                    else:
                        temp += "0,"
                    if x == "A" or x == "C":
                        temp += "1,"
                    else:
                        temp += "0,"
                    if x == "A":
                        count_AGCT[0] += 1
                        temp += str(round(count_AGCT[0] / position_mark * 1.0, 2))
                        temp += ','
                    elif x == "G":
                        count_AGCT[1] += 1
                        temp += str(round(count_AGCT[1] / position_mark * 1.0, 2))
                        temp += ','
                    elif x == "C":
                        count_AGCT[2] += 1
                        temp += str(round(count_AGCT[2] / position_mark * 1.0, 2))
                        temp += ','
                    elif x == gene_value:
                        count_AGCT[3] += 1
                        temp += str(round(count_AGCT[3] / position_mark * 1.0, 2))
                        temp += ','
                    elif x == "N":
                        count_AGCT[4] += 1
                        temp += str(round(count_AGCT[4] / position_mark * 1.0, 2))
                        temp += ','

                finnaly_lines += ((temp[:len(temp) - 1]) + '\n')
                # finally_text.write(temp+'\n')
        finally_text.writelines(finnaly_lines)
        return finally_text
        finally_text.close()


    def convert_without(dataPath, outputPath):
        """RFH feature"""
        lines = open(dataPath).readlines()
        finally_text = open(features_extraction+outputPath, 'w')
        finnaly_lines = ""
        for line in lines:
            if line.strip() == "": continue
            if line.strip()[0] in ['A', 'G', 'C', gene_value]:
                position_mark = 0
                count_AGCT = [0, 0, 0, 0]
                temp = ""
                re = []
                for i in line.strip():
                    re.append(i)
                #for x in list(line.strip()):
                for x in re:
                    position_mark += 1
                    if x == "A" or x == "G":
                        temp += "1,"
                    else:
                        temp += "0,"
                    if x == "A" or x == gene_value:
                        temp += "1,"
                    else:
                        temp += "0,"
                    if x == "A" or x == "C":
                        temp += "1,"
                    else:
                        temp += "0,"
                    if x == "A":
                        count_AGCT[0] += 1
                        temp += str(round(count_AGCT[0] / position_mark * 1.0, 2))
                        temp += ','
                    elif x == "G":
                        count_AGCT[1] += 1
                        temp += str(round(count_AGCT[1] / position_mark * 1.0, 2))
                        temp += ','
                    elif x == "C":
                        count_AGCT[2] += 1
                        temp += str(round(count_AGCT[2] / position_mark * 1.0, 2))
                        temp += ','
                    elif x == gene_value:
                        count_AGCT[3] += 1
                        temp += str(round(count_AGCT[3] / position_mark * 1.0, 2))
                        temp += ','

                finnaly_lines += ((temp[:len(temp) - 1]) + '\n')
                # finally_text.write(temp+'\n')
        finally_text.writelines(finnaly_lines)
        finally_text.close()
        return finally_text

    RFH_features_data = []
    if fill_NA == "1":
        RFH_features_data = convert_with(path + input_data, path + outputname)
        data = pd.read_csv(path + outputname, header=None, index_col=False)
        RFH_features_data = data.values
        print(data.values.shape)
    elif fill_NA == "0":
        RFH_features_data = convert_without(path + input_data, path + outputname)
        data = pd.read_csv(features_extraction + outputname, header=None, index_col=False)
        RFH_features_data = data.values
        print(data.values.shape)


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

    ###============================KNN=================================
    print("KNN")
    output_name= input_data.split(".")[0]+"_"+"KNN.csv"
    def read_fasta_file(path):
        fh = open(m6a_benchmark_dataset)
        seq = []
        for line in fh:
            if line.startswith('>'):
                continue
            else:
                seq.append(line.replace('\r', '').replace('\n', ''))
        fh.close()
        #matrix_data = np.array([list(e) for e in seq])
        matrix_data = []
        for e in seq:
            temp = []
            for i in e:
                temp.append(i)
            matrix_data.append(temp)
        return matrix_data

    def Comparing_score(query_sequence, original_sequence):
        score = 0
        for index, data in enumerate(query_sequence):
            if data == original_sequence[index]:
                score = score + 2
            else:
                score = score - 1
        return score

    def generating_one_column(matrix_data):
        whole_comparison_score = []
        the_begin_of_index = len(matrix_data) / 2
        for index_1, data_1 in enumerate(matrix_data):
            one_line_comparison_score = np.zeros(len(matrix_data))
            mark_origin_label = np.zeros(len(matrix_data))
            for index_2, data_2 in enumerate(matrix_data):
                if index_1 != index_2:
                    one_line_comparison_score[index_1] = -100
                    if index_2 < the_begin_of_index:
                        mark_origin_label[index_2] = 1
                    one_line_comparison_score[index_2] = Comparing_score(data_1, data_2)
            temp = []
            temp = [one_line_comparison_score, mark_origin_label]
            whole_comparison_score.append(temp)
        return whole_comparison_score

    def generating_features(matrix_data, K_list):
        matrix = generating_one_column(matrix_data)
        print np.asarray(matrix).shape
        whole_ = []
        for index, K_data in enumerate(K_list):
            line = []
            for data in matrix:
                idx = np.argsort(data[0])[::-1]
                idx = idx[xrange(K_data)]
                data[1] = pd.DataFrame(data[1])
                datas = data[1].iloc[idx]
                datas = datas.values
                line.append(sum(datas) / K_data)
            whole_.append(line)
        whole_ = np.array(whole_).T
        return whole_

    m6a_benchmark_dataset = input_data
    matrix_data = read_fasta_file(m6a_benchmark_dataset)
    KNN_features_data = generating_features(matrix_data, xrange(10, 201))
    print np.array(KNN_features_data).shape
    pd.DataFrame(KNN_features_data[0]).to_csv(features_extraction+output_name, header=None, index=False)

    ###============================MMI=================================
    print("MMI")
    import math
    path=""
    outputname=input_data.split(".")[0]+"_"+"MMI.csv"
    sequence=open(path+input_data)
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
            I_three[three_chars_index] = first_expression + second_expression - third_expression
        I_whole.extend(I_three)
        I_whole.extend(I_two)
        I_whole.extend(frequency_one_nolamdon)
        results.append([round(e,8) for e in I_whole])
    MMI_features_data = []
    MMI_features_data = results
    print(np.array(MMI_features_data).shape)
    pd.DataFrame(results).to_csv(features_extraction+outputname, header=None, index=None)
    sequence.close()

    ###============================PCP=================================
    print("PCP")
    propertyname="physical_chemical_properties_RNA_without.txt"
    if fill_NA=="1" and gene_type=="RNA":
        propertyname="physical_chemical_properties_RNA_without.txt"
    elif fill_NA=="0" and gene_type=="RNA":
        propertyname="physical_chemical_properties_RNA.txt"
    elif fill_NA=="1" and gene_type=="DNA":
        propertyname="physical_chemical_properties_DNA_without.txt"
    elif fill_NA=="0" and gene_type=="DNA":
        propertyname="physical_chemical_properties_DNA.txt"


    outputname=input_data.split(".")[0]+"_"+"PCP.csv"
    seq=[]
    physical_chemical_properties_path=path+propertyname
    m6a_2614_sequence=path+input_data

    fh=open(m6a_2614_sequence)
    for line in fh:#get the fasta sequence
        if line.startswith('>'):
            pass
        else:
            seq.append(line.replace('\n','').replace('\r',''))
    fh.close()

    data=pd.read_csv(physical_chemical_properties_path,header=None,index_col=None)#read the phisical chemichy proporties
    prop_key=data.values[:,0]

    if fill_NA=="1":
        prop_key[21]='NA'
    prop_data=data.values[:,1:]
    prop_data=np.matrix(prop_data)
    DNC_value=np.array(prop_data).T
    DNC_value_scale=[[]]*len(DNC_value)
    for i in xrange(len(DNC_value)):
        average_=sum(DNC_value[i]*1.0/len(DNC_value[i]))
        std_=np.std(DNC_value[i],ddof=1)
        DNC_value_scale[i]=[round((e-average_)/std_,2) for e in DNC_value[i]]
    prop_data_transformed=zip(*DNC_value_scale)
    # prop_data_transformed=StandardScaler().fit_transform(prop_data)
    prop_len=len(prop_data_transformed[0])

    whole_m6a_seq=seq
    i=0
    phisical_chemichy_len=len(prop_data_transformed)#the length of properties
    sequence_line_len=len(seq[0])#the length of one sequence
    LAMDA=4
    PCP_features_data=[]#used to save the fanal result
    for one_m6a_sequence_line in whole_m6a_seq:
        one_sequence_value=[[]]*(sequence_line_len-1)
        PC_m=[0.0]*prop_len
        PC_m=np.array(PC_m)
        for one_sequence_index in range(sequence_line_len-1):
            for prop_index in xrange(len(prop_key)):
                if one_m6a_sequence_line[one_sequence_index:one_sequence_index+2]==prop_key[prop_index]:
                    one_sequence_value[one_sequence_index]=prop_data_transformed[prop_index]
            PC_m+=np.array(one_sequence_value[one_sequence_index])
        PC_m=PC_m/(sequence_line_len-1)
        auto_value=[]
        for LAMDA_index in xrange(1,LAMDA+1):
            temp = [0.0] * prop_len
            temp=np.array(temp)
            for auto_index in xrange(1,sequence_line_len-LAMDA_index):
                temp=temp+(np.array(one_sequence_value[auto_index-1])-PC_m)*(np.array(one_sequence_value[auto_index+LAMDA_index-1])-PC_m)
                temp=[round(e,8) for e in temp.astype(float)]
            x=[round(e/(sequence_line_len-LAMDA_index-1),8) for e in temp]
            auto_value.extend([round(e,8) for e in x])
        for LAMDA_index in xrange(1, LAMDA + 1):
            for i in xrange(1,prop_len+1):
                for j in xrange(1,prop_len+1):
                    temp2=0.0
                    if i != j:
                        for auto_index in xrange(1, sequence_line_len - LAMDA_index):
                                temp2+=(one_sequence_value[auto_index-1][i-1]-PC_m[i-1])*(one_sequence_value[auto_index+LAMDA_index-1][j-1]-PC_m[j-1])
                        auto_value.append(round(temp2/((sequence_line_len-1)-LAMDA_index),8))
        PCP_features_data.append(auto_value)
    print(np.array(PCP_features_data).shape)
    pd.DataFrame(PCP_features_data).to_csv(features_extraction+outputname,header=None,index=False)

    ###============================PseDNC=================================
    print("PseDNC")
    outputname=input_data.split(".")[0]+"_"+"PseDNC.csv"
    propertyname="physical_chemical_properties_3_RNA_without.txt"
    if fill_NA=="1" and gene_type=="RNA":
        propertyname="physical_chemical_properties_3_RNA_without.txt"
    elif fill_NA=="0" and gene_type=="RNA":
        propertyname="physical_chemical_properties_3_RNA.txt"
    elif fill_NA=="1" and gene_type=="DNA":
        propertyname="physical_chemical_properties_3_DNA_without.txt"
    elif fill_NA=="0" and gene_type=="DNA":
        propertyname="physical_chemical_properties_3_DNA.txt"
    print(propertyname)
    phisical_chemical_proporties=pd.read_csv(path+propertyname,header=None,index_col=None)
    m6a_sequence=open(path+input_data)
    DNC_key=phisical_chemical_proporties.values[:,0]

    if fill_NA=="1":
        DNC_key[21]='NA'
    DNC_value=phisical_chemical_proporties.values[:,1:]
    DNC_value=np.array(DNC_value).T
    DNC_value_scale=[[]]*len(DNC_value)
    for i in xrange(len(DNC_value)):
        average_=sum(DNC_value[i]*1.0/len(DNC_value[i]))
        std_=np.std(DNC_value[i],ddof=1)
        DNC_value_scale[i]=[round((e-average_)/std_,2) for e in DNC_value[i]]
    DNC_value_scale=zip(*DNC_value_scale)


    DNC_len=len(DNC_value_scale[0])
    m6aseq=[]
    for line in m6a_sequence:
        if line.startswith('>'):
            pass
        else:
            m6aseq.append(line.replace('\n','').replace("\r",''))
    w=0.9
    Lamda=6
    result_value=[]
    m6a_len=len(m6aseq[0])
    m6a_num=len(m6aseq)
    for m6a_line_index in xrange(m6a_num):
        frequency=[0]*len(DNC_key)
        m6a_DNC_value=[[]]*(m6a_len-1)
        for m6a_line_doublechar_index in xrange(m6a_len):
            for DNC_index in xrange(len(DNC_key)):
                if m6aseq[m6a_line_index][m6a_line_doublechar_index:m6a_line_doublechar_index+2]==DNC_key[DNC_index]:
                    m6a_DNC_value[m6a_line_doublechar_index]=DNC_value_scale[DNC_index]
                    frequency[DNC_index]+=1
        frequency=[e/float(sum(frequency)) for e in frequency]
        p=sum((frequency))
        #frequency=np.array(frequency)/float(sum(frequency))#(m6a_len-1)
        one_line_value_with = 0.0
        sita = [0] * Lamda
        for lambda_index in xrange(1, Lamda + 1):
            one_line_value_without_ = 0.0
            for m6a_sequence_value_index in xrange(1, m6a_len - lambda_index):
                temp = map(lambda (x, y): round((x - y) ** 2,8), zip(np.array(m6a_DNC_value[m6a_sequence_value_index - 1]), np.array(m6a_DNC_value[m6a_sequence_value_index - 1 + lambda_index])))
                temp_value = round(sum(temp) * 1.0 / DNC_len,8)
                one_line_value_without_ += temp_value
            one_line_value_without_ = round(one_line_value_without_ / (m6a_len - lambda_index-1),8)
            sita[lambda_index - 1] = one_line_value_without_
            one_line_value_with += one_line_value_without_
        dim = [0] * (len(DNC_key) + Lamda)
        for index in xrange(1, len(DNC_key) + Lamda+1):
            if index <= len(DNC_key):
                dim[index - 1] = frequency[index - 1] / (1.0 + w * one_line_value_with)
            else:
                dim[index - 1] = w * sita[index - len(DNC_key)-1] / (1.0 + w * one_line_value_with)
            dim[index-1]=round(dim[index-1],8)
        result_value.append(dim)
    print(np.array(result_value).shape)
    pd.DataFrame(result_value).to_csv(features_extraction+outputname, header=None, index=None)
    m6a_sequence.close()




