#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

# In[ ]:


def ModelTest(data_name,interval) :
    
    data = pd.read_csv("{}".format(data_name),header=None)
    test_cnt=len(data.iloc[:,0])/interval
    data.columns=['Math Expresion','Expression Name']
    data.loc[data["Expression Name"]=='distributor_expression','Expression Name']=0
    data.loc[data["Expression Name"]=='linear_expression','Expression Name']=1
    data.loc[data["Expression Name"]=='linear_equation','Expression Name']=2
    
    cnt=0
    for s in data.iloc[:,0]:
        data.iloc[cnt,0] = bytearray(s, 'utf-8')
        cnt+=1
        
    start = 0
    i_cnt = interval
    earlystop = EarlyStopping(monitor='val_loss', mode='min',min_delta=0.0001, verbose = 1, patience=5)
    
    epoch_cnt=[]
    best_loss=[]
    data_cnt=[]
    best_score=[]
    
    while(test_cnt>0):
        
        start += 1
        print("   {}번 구간 시작".format(start))
        print()
        mc = ModelCheckpoint('./models/best_model_{}.h5'.format(i_cnt), monitor='val_loss', mode='min', save_best_only=True)
        data_i = data.iloc[:i_cnt,].copy()            
        X_train,X_test,y_train,y_test = train_test_split(data_i.iloc[:,0],data_i.iloc[:,1],test_size=0.2)
        X_train_seq = sequence.pad_sequences(X_train, maxlen=200)
        X_test_seq = sequence.pad_sequences(X_test, maxlen=200)
        Y_train_ctg = np_utils.to_categorical(y_train)
        Y_test_ctg = np_utils.to_categorical(y_test)
        model = Sequential()
        model.add(Embedding(len(X_train), 100))
        model.add(LSTM(100, activation='tanh'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        history = model.fit(X_train_seq, Y_train_ctg, batch_size=128, 
                        epochs=200,validation_data=(X_test_seq, Y_test_ctg),callbacks=[earlystop,mc])
        print("\n Test Accuracy: %.4f" % (model.evaluate(X_test_seq, Y_test_ctg)[1]))
        
        data_cnt.append(i_cnt)
        epoch_cnt.append(history.epoch[-5])
        best_loss.append(min(history.history['val_loss']))
        best_score.append(max(history.history['val_accuracy']))
        
        test_cnt-=1
        
        i_cnt+=interval
        
        
        
        print("   {}번 구간 완료".format(start))
        print()
        print()
        
    result = pd.DataFrame({'학습데이터량':data_cnt,
                           'Epoch' : epoch_cnt,
                           '손실최소값':best_loss,
                          '정확도' :best_score })
    
    return result

def SaveModel(model,data_name) :
    model.save('{}.h5'.format(data_name))
    

def LoadModel(model_name) :
    model = load_model('{}.h5'.format(model_name))
    return model

