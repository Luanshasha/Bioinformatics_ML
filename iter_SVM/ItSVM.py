# -*- coding: utf-8 -*-

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
def getopt():
    '''parsing the opts'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help="The path of genome file,should be a fasta file",required=True)
    parser.add_argument('-t', '--times', type=int, help="iterations", required=True)
    parser.add_argument('-n', '--cv_number', type=int, help="fold number", required=True)
    parser.add_argument('-c', '--CPU_value', type=int, help="the number of cup  you will use")
    args = parser.parse_args()
    return args
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
    SN = TP/(TP + FN) #Sensitivity = TP/P  and P = TP + FN
    SP = TN/(FP + TN) #Specificity = TN/N  and N = TN + FP
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    GM=math.sqrt(recall*SP)
    #MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return precision,recall,SN,SP,GM,TP,TN,FP,FN

def SVM_calssfier(input_feature,proba_dir,model_dir,result_dir,crossvalidation_values,CPU_values,output_name,t):

    X=input_feature
    Y=list(map(lambda x:1,xrange(len(input_feature)//2)))
    Y2 = list(map(lambda x: 0, xrange(len(input_feature) // 2)))
    Y.extend(Y2)
    Y = np.array(Y)
    d = input_feature.shape[1]
    svc = svm.SVC(probability=True)
    parameters = {'kernel': ['rbf'], 'C':map(lambda x:2**x,np.linspace(-2,5,7)), 'gamma':map(lambda x:2**x,np.linspace(-5,2,7))}
    clf = GridSearchCV(svc, parameters, cv=crossvalidation_values, n_jobs=CPU_values, scoring='accuracy')
    clf.fit(X,Y)
    C=clf.best_params_['C']
    gamma=clf.best_params_['gamma']
    y_predict=cross_val_predict(svm.SVC(kernel='rbf',C=C,gamma=gamma),X,Y,cv=crossvalidation_values,n_jobs=CPU_values)
    y_predict_prob=cross_val_predict(svm.SVC(kernel='rbf',C=C,gamma=gamma,probability=True),X,Y,cv=crossvalidation_values,n_jobs=CPU_values,method='predict_proba')

    joblib.dump(clf,model_dir+str(t)+"time.model")
    predict_save=[y_predict_prob[:,0],y_predict_prob[:,1]]
    predict_save=np.array(predict_save).T
    pd.DataFrame(predict_save).to_csv(proba_dir+str(t)+'time.csv',header=None,index=False)
    ROC_AUC_area=metrics.roc_auc_score(Y,y_predict_prob[:,1])
    ACC=metrics.accuracy_score(Y,y_predict)
    precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(Y, y_predict)
    F1_Score=metrics.f1_score(Y, y_predict)
    F_measure=F1_Score
    MCC=metrics.matthews_corrcoef(Y, y_predict)
    pos=TP+FN
    neg=FP+TN
    savedata=[[['SVM'+"C:"+str(C)+"gamma:"+str(gamma),ACC,precision, recall,SN, SP, GM,F_measure,F1_Score,MCC,ROC_AUC_area,TP,FN,FP,TN,pos,neg]]]
    easy_excel.save(classifier+"_crossvalidation",[str(X.shape[1])],savedata,result_dir+'itSVM'+output_name+'cross_validation_'+str(d)+'D_'+str(t)+'time.xls')


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
    file_path_list=list()
    print file_name_list
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
    
#### 归一化可以不用，也可以用，都跑下结果试试
def feature_scaler(feature):
    scaler_model=MinMaxScaler()
    scaler_model.fit(feature)
    scaler_feature=scaler_model.transform(feature)
    return scaler_feature,scaler_model

def run(input_feature,proba_dir,model_dir,scaler_model_dir,result_dir,crossvalidation_values,CPU_values,output_name,n):
    for i in range(n):
        print i
        if i==0:
            SVM_calssfier(input_feature,proba_dir,model_dir,result_dir,crossvalidation_values,CPU_values,output_name,i+1)
            i=i+1
        else:
            proba_feature=concat_proba_feature(proba_dir)
            feature=concat_feature(input_feature,proba_feature)
            feature,scaler_model=feature_scaler(feature)
            joblib.dump(scaler_model,os.path.join(scaler_model_dir,str(i+1)+'time_scaler.pkl'))
            SVM_calssfier(feature,proba_dir,model_dir,result_dir,crossvalidation_values,CPU_values,output_name,i+1)
            i=i+1

if __name__=='__main__':
    args=getopt()
    proba_dir='./proba/'
    scaler_model_dir='./scaler_model/'
    model_dir='./model/'
    result_dir='./result/'
    crossvalidation_values = args.cv_number
    CPU_values =args.CPU_value
    classifier = "SVM"
    mode = "crossvalidation"
    input_path =args.input
    output_name=input_path.split('.')[0]
    commands.getoutput('mkdir proba')
    commands.getoutput('mkdir scaler_model')
    commands.getoutput('mkdir result')
    commands.getoutput('mkdir model')
    iter_n=args.times
    input_feature=read_input_feature(input_path)
    run(input_feature,proba_dir,model_dir,scaler_model_dir,result_dir,crossvalidation_values,CPU_values,output_name,iter_n)
