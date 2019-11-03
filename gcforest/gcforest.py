"""
MNIST datasets demo for gcforest
Usage:
    define the model within scripts:
        python examples/demo_mnist.py
    get config from json file:
        python examples/demo_mnist.py --model examples/demo_mnist-gc.json
        python examples/demo_mnist.py --model examples/demo_mnist-ca.json
"""

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import argparse
import numpy as np
#np.set_printoptions(threshold=np.inf)
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import math
from keras.datasets import mnist
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
sys.path.insert(0, "lib")
from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json
import easy_excel

inputdata = sys.argv[1]
outputname=inputdata.split(".")[0]
K_fold = int(sys.argv[2])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, default=None, help="gcfoest Net Model File")
    parser.add_argument("-i",type=str,help="train file")
    parser.add_argument("-n" ,type=int,help="Number of cross validations")
    parser.add_argument("-o", type=str,help="result file")
    args = parser.parse_args()
    return args


def get_toy_config(n):
    config = {}
    ca_config = {}
    #ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = n
    ca_config["estimators"] = []
    ca_config["estimators"].append(
            {"n_folds": K_fold, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
             "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1,"num_class": n} )
    ca_config["estimators"].append({"n_folds": K_fold, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": K_fold, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": K_fold, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config
 



    
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

from numpy.random import seed
seed(1)
#from tensorflow import set_random_seed
#set_random_seed(1)
if __name__ == "__main__":
    '''args = parse_args()
    inputdata = args.i
    K_fold = args.n
    outputname = args.o '''
    
    classifier = "GCFOREST"
    mode = "crossvalidation"
    print("start")
    train_data = pd.read_csv(inputdata, header=None, index_col=None)
    print(len(train_data))
    X_train = np.array(train_data)
    print(X_train.shape)
    Y = list(map(lambda x: 1, range(len(train_data) // 2)))
    Y2 = list(map(lambda x: 0, range(len(train_data) // 2)))
    Y.extend(Y2)
    Y_train = np.array(Y)

    '''X_train = X[:,1:]
    print(X_train.shape)
    Y_train =X[:,0]'''
    #length = len(X_train)/K_fold
    '''print(Y_train.shape)
    np.random.seed(116)
    np.random.shuffle(X_train)
    np.random.seed(116)
    np.random.shuffle(Y_train)'''
    label_set = list(set(list(Y_train)))
    label_set.sort(key=list(Y_train).index)
    #if args.model is None:
    config = get_toy_config(len(label_set))
    #else:
        #config = load_json(args.model)

    gc = GCForest(config)
    # If the model you use cost too much memory for you.
    # You can use these methods to force gcforest not keeping model in memory
    # gc.set_keep_model_in_mem(False), default is TRUE.
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    
    '''i=0
    length = len(X_train)/K_fold
    j=length
    ACC=[]
    y_predict_all=[]
    y_test_all=[]
    y_predict_prob_all=[]
    for k in range(K_fold):
        test = X_train[int(i):int(j)]
        y_test = Y_train[int(i):int(j)]
        train = np.append(X_train[0:int(i)],X_train[int(j):],axis=0)
        y_train = np.append(Y_train[0:int(i)],Y_train[int(j):],axis=0)
        
        y_test_all.extend(list(y_test))
        gc.fit_transform(train, y_train)
        y_predict = gc.predict(test)
        y_predict_all.extend(list(y_predict))
        
        y_predict_prob = gc.predict_proba(test)[:,1]
        y_predict_prob_all.extend(list(y_predict_prob))
        i+=length
        j+=length'''
    
    gc.fit_transform(X_train, Y_train)    
    y_predict = gc.predict(X_test)
    y_predict_prob = gc.predict_proba(X_test)[:,1]
    acc = accuracy_score(Y_test, y_predict)
    print("Test Accuracy of GcForest (save and load) = {:.2f} %".format(acc * 100))
    ROC_AUC_area=metrics.roc_auc_score(Y_test,y_predict_prob)
    print("ROC ="+str(ROC_AUC_area))
    ACC=metrics.accuracy_score(Y_test,y_predict)
    print("ACC:"+str(ACC))
    precision, recall, SN, SP, GM, TP, TN, FP, FN = performance(Y_test, y_predict)
    F1_Score=metrics.f1_score(Y_test, y_predict)
    F_measure=F1_Score
    MCC=metrics.matthews_corrcoef(Y_test, y_predict)
    pos=TP+FN
    neg=FP+TN
    savedata=[[['gcforest',ACC,precision, recall,SN, SP, GM,F_measure,F1_Score,MCC,ROC_AUC_area,TP,FN,FP,TN,pos,neg]]]
    easy_excel.save(classifier+"_crossvalidation",[str(X_train.shape[1])],savedata,'cross_validation_'+classifier+"_"+outputname+'.xls')
    
   
