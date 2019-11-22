import argparse
import pandas as pd
import numpy as np
from sklearn import svm
import math
import easy_excel
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn import metrics
import sys
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
import os
import commands

def performance(labelArr, predictArr):
    #labelArr[i] is actual value,predictArr[i] is predict value
    TP = 0.; TN = 0.; FP = 0.; FN = 0.
    for i in range(len(labelArr)):
        if labelArr[i] == 1 and predictArr[i] == 1:
            TP += 1.
        if labelArr[i] == 1 and predictArr[i] == 0:
            FN += 1.
        if labelArr[i] == 0 and predictArr[i] == 1:
            FP += 1.
        if labelArr[i] == 0 and predictArr[i] == 0:
            TN += 1.
    if (TP + FN)==0:
        SN=0
    else:
        SN = TP/(TP + FN) #Sensitivity = TP/P  and P = TP + FN
    if (FP+TN)==0:
        SP=0
    else:
        SP = TN/(FP + TN) #Specificity = TN/N  and N = TN + FP
    if (TP+FP)==0:
        precision=0
    else:
        precision=TP/(TP+FP)
    if (TP+FN)==0:
        recall=0
    else:
        recall=TP/(TP+FN)
    GM=math.sqrt(recall*SP)
    #MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return precision,recall,SN,SP,GM,TP,TN,FP,FN
def concat_feature(input_feature,proba_feature):
    input_feature = pd.DataFrame(input_feature).astype(float)
    proba_feature = pd.DataFrame(proba_feature).astype(float)
    feature=pd.concat([input_feature,proba_feature],axis=1)
    return feature

def read_input_feature(path):
    input_feature=pd.read_csv(path,header=None,index_col=None)
    input_feature=np.array(input_feature)
    return input_feature


def concat_proba_feature(path):
    file_name_list=os.listdir(path)
    file_name_list.sort(key= lambda x:int(x[:-8]))
    print file_name_list
    file_path_list=list()
    for file_name in file_name_list:
        file_path_list.append(os.path.join(path,file_name))
    proba_file_list=[]
    for proba_file in file_path_list:
        proba_reader=pd.read_csv(proba_file,header=None,index_col=None)
        proba_file_list.append(proba_reader)
    proba_feature_save=[]
    for proba_feature in proba_file_list:
        proba_feature=np.array(proba_feature)
        proba_feature_save.append(proba_feature[:,0])
        proba_feature_save.append(proba_feature[:,1])
    proba_feature_save=np.array(proba_feature_save).T
    #print proba_feature_save.shape
    return proba_feature_save
def SVM_test(input_feature,svm_model,test_proba_dir,test_result_dir,output_name,i):
    X=input_feature
    Y = list(map(lambda x: 1, xrange(len(input_feature) // 2)))
    Y2 = list(map(lambda x: 0, xrange(len(input_feature) // 2)))
    Y.extend(Y2)
    Y = np.array(Y)
    d = input_feature.shape[1]

    X_predict = svm_model.predict(X)
    X_predict_proba = svm_model.predict_proba(X)
    predict_save = [X_predict_proba[:, 0], X_predict_proba[:, 1]]
    predict_save = np.array(predict_save).T
    pd.DataFrame(predict_save).to_csv(test_proba_dir+str(i)+'time.csv', header=None, index=False)
    ROC_AUC_area = metrics.roc_auc_score(Y, X_predict_proba[:, 1])
    ACC = metrics.accuracy_score(Y, X_predict)
    precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(Y, X_predict)
    F1_Score = metrics.f1_score(Y, X_predict)
    F_measure = F1_Score
    MCC = metrics.matthews_corrcoef(Y, X_predict)
    pos = TP + FN
    neg = FP + TN
    C = svm_model.best_params_['C']
    gamma = svm_model.best_params_['gamma']
    savedata = [[['SVM' + "C:" + str(C) + "gamma:" + str(gamma), ACC, precision, recall, SN, SP, GM, F_measure,
                  F1_Score, MCC, ROC_AUC_area, TP, FN, FP, TN, pos, neg]]]
    easy_excel.save("SVM_crossvalidation", [str(X.shape[1])], savedata,
                    test_result_dir+output_name+'_'+str(d)+'_D'+str(i)+'time.xls')



def feature_scaler(feature,scaler_model):
    scaler_feature=scaler_model.transform(feature)
    return scaler_feature


def run (input_feature,test_proba_dir,test_result_dir,output_name,scaler_model_dir,model_dir,n):
    for i in range(n):
        if i==0:
            d=input_feature.shape[1]
            svm_model=joblib.load(os.path.join(model_dir,str(i+1)+'time.model'))
            SVM_test(input_feature,svm_model,test_proba_dir,test_result_dir,output_name,i+1)
            i=i+1
        else:
            proba_feature=concat_proba_feature(test_proba_dir)
            feature = concat_feature(input_feature, proba_feature)
            d=feature.shape[1]
            scaler_model=joblib.load(os.path.join(scaler_model_dir,str(i+1)+'time_scaler.pkl'))
            feature=feature_scaler(feature,scaler_model)
            svm_model = joblib.load(
                os.path.join(model_dir, str(i + 1) + 'time.model'))
            SVM_test(feature, svm_model, test_proba_dir, test_result_dir, output_name, i + 1)

def getopt():
    '''parsing the opts'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help="The path of genome file,should be a fasta file",required=True)
    parser.add_argument('-t', '--times', type=int, help="iterations", required=True)
    args = parser.parse_args()
    return args
if __name__=="__main__":
    args=getopt()
    input_path=args.input
    output_name=input_path.split('.')[0]
    time=args.times
    input_feature=read_input_feature(input_path)
    model_dir='./model/'
    scaler_model_dir = './scaler_model/'
    test_proba_dir='./test_proba/'
    test_result_dir='./test_result/'
    commands.getoutput('mkdir test_proba')
    commands.getoutput('mkdir test_result')
    run(input_feature, test_proba_dir, test_result_dir, output_name, scaler_model_dir, model_dir, time)



