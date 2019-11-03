"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import resnet_change
import sys
import keras
import pandas as pd
import random
from sklearn.utils import shuffle
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import os
from sklearn.model_selection import StratifiedKFold

input_data = sys.argv[1]
output_name = sys.argv[2]#'resnet_18_new_celegans'
#output_results =sys.argv[2]
#flatten_name = output_results.split(".")[0]
#batch_size_num = int(sys.argv[3])
#resnet_name = sys.argv[3] ###输入resnet的名字。例如“resnet18”
model_name = sys.argv[3]#"resnet_18_celegans_best_model.h5"#sys.argv[4]
#model_name = modelpath.split('.')[0]




#csv_logger = CSVLogger('resnet18_cifar10.csv')
batch_size_num = 4
#batch_size_num = [4, 8, 16, 32]
#nb_classes = 10
#nb_classes = 2 ###[1,0]代表1，为正例，[0,1]代表0，为负例
nb_classes = 1
nb_epoch = 200
data_augmentation = False

# input image dimensions
img_rows, img_cols = 41, 4

X = np.load(file=input_data)
print(X.shape)
Y = list(map(lambda x: 1, range(len(X) // 2)))
Y2 = list(map(lambda x: 0, range(len(X) // 2)))
Y.extend(Y2)
Y = np.array(Y)
print(Y.shape)

'''X_train,X_test_val,Y_train,Y_test_val = train_test_split(X,Y,test_size=0.4, random_state=32)###按照行随机打乱的，random_state固定保证每次数据打乱的都是一样的顺序
X_test,X_val,Y_test,Y_val = train_test_split(X_test_val,Y_test_val,test_size=0.5, random_state=32)
np.save(input_data.split(".")[0] + "_Test_data.npy",X_val)
np.save(input_data.split(".")[0] + "_Train.npy",X_train)
pd.DataFrame(Y_val).to_csv(input_data.split('.')[0]+"_Test_label.csv",header=None,index=None)
pd.DataFrame(Y_train).to_csv(input_data.split('.')[0]+"_Train_label.csv",header=None,index=None)
'''
#model.summary()

###提取中间层（flatten_1）的特征
'''def extract_layer_feature(base_model,x,y):
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten_1').output)
    flatten_1_data = model.predict(x)
    flatten_1_label  = model.predict(y)
    pd.DataFrame(flatten_1_data).to_csv(input_data.split(".")[0] + "_" + resnet_name + "_" + "flatten_1_data.csv" )
    pd.DataFrame(flatten_1_label).to_csv(input_data.split(".")[0] + "_" + resnet_name + "_" + "flatten_1_label.csv" )
    #return flatten_1_data,flatten_1_label
'''
kfold = StratifiedKFold(n_splits = 5,shuffle = True, random_state = 32)
index = 1
for train, test in kfold.split(X, Y):
    model = resnet_change.ResnetBuilder.build_resnet_18((img_rows, img_cols), nb_classes)  # img_channels,
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    X_train = X[train]
    Y_train = Y[train]
    X_test = X[test]
    Y_test = Y[test]
    checkpoint = ModelCheckpoint(model_name + "_" + str(index) + ".h5", monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max', period=1)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger(output_name + "_" + str(index) + ".csv")

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                    batch_size=batch_size_num,
                    nb_epoch=200,
                    validation_data=(X_test, Y_test),
                    shuffle=True,
                    callbacks=[lr_reducer, early_stopper, csv_logger, checkpoint])
        #extract_layer_feature(model,X_train)
        #model.save(model_name + "_" + str(index) + ".h5")
        index += 1
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size_num),
                            steps_per_epoch=X_train.shape[0] // batch_size_num,
                            validation_data=(X_test, Y_test),
                            epochs=nb_epoch, verbose=1, max_q_size=100,
                            callbacks=[lr_reducer, early_stopper, csv_logger,checkpoint])
        #extract_layer_feature(model,X_train)
        #model.save(model_name)

