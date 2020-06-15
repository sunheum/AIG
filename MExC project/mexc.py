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
import time
import matplotlib.pyplot as plt
import warnings

# 경고 메세지 무시
warnings.filterwarnings('ignore')

# tensorflow-gpu 사용시 메모리 관련 에러를 잡아줌
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

model=load_model('./models/best_model_default.h5')

# In[2]:

def mexc_pre(math_expression):
    """
    
    수학 수식을 입력받아 수식명을 예측해줍니다.
    
    :param math_expresstion: 수식데이터 파일명(수식데이터는 latex로 구성된 csv 형식의 파일이어야함)
    :return pre_list: 수식명을 예측한 리스트(리스트의 요소는 str형식) 
    
    """
    #수식파일을 dataframe 형식으로 불러옵니다.
    data = pd.read_csv("{}".format(math_expression),header=None)
    
    #수식들을 lstm 모델이 학습할 수 있도록 바이트 코드로 변환해줍니다
    num=0
    for s in data.iloc[:,0]:
        data.iloc[num,0] = bytearray(s, 'utf-8')
        num+=1
        
    X_test = sequence.pad_sequences(data.iloc[:,0], maxlen=200)
    
    #모델에 수식을 입력하고 예측값을 뽑아냅니다
    pre = model.predict(X_test)
    
    #예측값을 문자로 치환해줍니다
    
    pre_list=[]
    for i in range(len(pre)) :
        if np.argmax(pre[i]) == 0 :
            pre_list.append('distributor_expression')
        elif np.argmax(pre[i]) == 1 :
            pre_list.append('linear_expression')
        elif np.argmax(pre[i]) == 2 :
            pre_list.append('linear_equation')
        elif np.argmax(pre[i]) == 3 :
            pre_list.append('inequality_equation')
        elif np.argmax(pre[i]) == 4 :
            pre_list.append('frac_multiplication_expression')
        elif np.argmax(pre[i]) == 5 :
            pre_list.append('frac_division_expression')
        elif np.argmax(pre[i]) == 6 :
            pre_list.append('exponential_expression_1')
        elif np.argmax(pre[i]) == 7 :
            pre_list.append('exponential_expression_1_1')
        elif np.argmax(pre[i]) == 8 :
            pre_list.append('exponential_expression_2')
        elif np.argmax(pre[i]) == 9 :
            pre_list.append('exponential_expression_3')
        elif np.argmax(pre[i]) == 10 :
            pre_list.append('exponential_expression_4')
        elif np.argmax(pre[i]) == 11 :
            pre_list.append('frac_exponential_expression_1')
        elif np.argmax(pre[i]) == 12 :
            pre_list.append('frac_exponential_expression_2')
        elif np.argmax(pre[i]) == 13 :
            pre_list.append('frac_exponential_expression_3')
        elif np.argmax(pre[i]) == 14 :
            pre_list.append('frac_exponential_expression_4')
        elif np.argmax(pre[i]) == 15 :
            pre_list.append('variable_expression1')
        elif np.argmax(pre[i]) == 16 :
            pre_list.append('variable_expression2')
        elif np.argmax(pre[i]) == 17 :
            pre_list.append('solution_equation')
        elif np.argmax(pre[i]) == 18 :
            pre_list.append('variable')
            
            
    
    return pre_list
    
    
def change_model(model_name) :
    """
    
    MExC를 처음 로드 했을 때 불러오는 예측모델을 사용자가 설정하는 모델로 변경해줍니다
    
    :param model_name: 변경할 모델의 파일명
    
    """
    global model
    model = load_model('{}'.format(model_name))    
    

def model_fit(data_name,epoch,save_name):
    """
    수식과 수식명으로 이루어진 데이터를 입력받아 새로운 수식예측 모델을 생성합니다.
    
    :param data_name: 데이터 파일명(수식과 수식명이 csv 형식으로 이루어진 파일이어야 함)
    :param epoch: 학습횟수 설정
    :param save_name: 저장할 모델이름(이름 앞에 new_model_~이 붙는다)
    :return history:새로 생성된 모델의 학습 히스토리정보
    
    """
    data = pd.read_csv("{}".format(data_name),header=None)
    
    #불러온 데이터프레임에 header를 설정함
    data.columns=['Math Expresion','Expression Name']
        
    #one-hot 인코딩
    data.loc[data["Expression Name"]=='distributor_expression','Expression Name']=0
    data.loc[data["Expression Name"]=='linear_expression','Expression Name']=1
    data.loc[data["Expression Name"]=='linear_equation','Expression Name']=2
    data.loc[data["Expression Name"]=='inequality_equation','Expression Name']=3
    data.loc[data["Expression Name"]=='frac_multiplication_expression','Expression Name']=4
    data.loc[data["Expression Name"]=='frac_division_expression','Expression Name']=5
    data.loc[data["Expression Name"]=='exponential_expression_1','Expression Name']=6
    data.loc[data["Expression Name"]=='exponential_expression_1_1','Expression Name']=7
    data.loc[data["Expression Name"]=='exponential_expression_2','Expression Name']=8
    data.loc[data["Expression Name"]=='exponential_expression_3','Expression Name']=9
    data.loc[data["Expression Name"]=='exponential_expression_4','Expression Name']=10
    data.loc[data["Expression Name"]=='frac_exponential_expression_1','Expression Name']=11
    data.loc[data["Expression Name"]=='frac_exponential_expression_2','Expression Name']=12
    data.loc[data["Expression Name"]=='frac_exponential_expression_3','Expression Name']=13
    data.loc[data["Expression Name"]=='frac_exponential_expression_4','Expression Name']=14
    data.loc[data["Expression Name"]=='variable_expression1','Expression Name']=15
    data.loc[data["Expression Name"]=='variable_expression2','Expression Name']=16
    data.loc[data["Expression Name"]=='solution_equation','Expression Name']=17
    data.loc[data["Expression Name"]=='variable','Expression Name']=18
    
    num=0
    for s in data.iloc[:,0]:
        data.iloc[num,0] = bytearray(s, 'utf-8')
        num+=1
    
    #학습프로세스 진행중 callback 함수 인자값 설정
    earlystop = EarlyStopping(monitor='val_loss', mode='min',min_delta=0.0001, verbose = 1, patience=5)
    mc = ModelCheckpoint('./models/new_model_{}.h5'.format(save_name), monitor='val_loss', mode='min', save_best_only=True)
    
    #학습용데이터, 검증용데이터로 분할
    X_train,X_test,y_train,y_test = train_test_split(data.iloc[:,0],data.iloc[:,1],test_size=0.2)
    X_train_seq = sequence.pad_sequences(X_train, maxlen=200)
    X_test_seq = sequence.pad_sequences(X_test, maxlen=200)
    Y_train_ctg = np_utils.to_categorical(y_train)
    Y_test_ctg = np_utils.to_categorical(y_test)
    
    #모델 생성
    model = Sequential()
    model.add(Embedding(len(X_train), 100))
    model.add(LSTM(100, activation='tanh'))
    model.add(Dense(19, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    history = model.fit(X_train_seq, Y_train_ctg, batch_size=128, 
                        epochs=epoch,validation_data=(X_test_seq, Y_test_ctg),callbacks=[earlystop,mc])
    #모델 생성결과 출력
    if history.epoch == True:
        print("\n ====모델 생성완료====")
    else :
        print("\n ====모델 생성완료====")
       
    return history
    
def test_model(model_name,test_data) :
    """
    모델의 정확도를 테스트할 수 있다.
    
    :param model_name:테스트할 모델의 파일명
    :param test_data:테스트 테이터의 파일명(수식과 수식명이 csv 형식으로 이루어진 파일이어야 함)
    :return result:테스트 결과값(DataFrame 형식)
    
    """
    #모델을 불러오고 테스트 데이터를 전처리하는 과정
    model = load_model('{}'.format(model_name))
    test= pd.read_csv('{}'.format(test_data),header=None)
    test.columns=['Math Expresion','Expression Name']
    
    #result에 사용하기 위해 전처리 전에 원본데이터를 백업해준다
    test_ME = test.iloc[:,0].reset_index(drop=True)
    test_target = test.iloc[:,1].reset_index(drop=True)
    test.loc[test["Expression Name"]=='distributor_expression','Expression Name']=0
    test.loc[test["Expression Name"]=='linear_expression','Expression Name']=1
    test.loc[test["Expression Name"]=='linear_equation','Expression Name']=2
    test.loc[test["Expression Name"]=='inequality_equation','Expression Name']=3
    test.loc[test["Expression Name"]=='frac_multiplication_expression','Expression Name']=4
    test.loc[test["Expression Name"]=='frac_division_expression','Expression Name']=5
    test.loc[test["Expression Name"]=='exponential_expression_1','Expression Name']=6
    test.loc[test["Expression Name"]=='exponential_expression_1_1','Expression Name']=7
    test.loc[test["Expression Name"]=='exponential_expression_2','Expression Name']=8
    test.loc[test["Expression Name"]=='exponential_expression_3','Expression Name']=9
    test.loc[test["Expression Name"]=='exponential_expression_4','Expression Name']=10
    test.loc[test["Expression Name"]=='frac_exponential_expression_1','Expression Name']=11
    test.loc[test["Expression Name"]=='frac_exponential_expression_2','Expression Name']=12
    test.loc[test["Expression Name"]=='frac_exponential_expression_3','Expression Name']=13
    test.loc[test["Expression Name"]=='frac_exponential_expression_4','Expression Name']=14
    test.loc[test["Expression Name"]=='variable_expression1','Expression Name']=15
    test.loc[test["Expression Name"]=='variable_expression2','Expression Name']=16
    test.loc[test["Expression Name"]=='solution_equation','Expression Name']=17
    test.loc[test["Expression Name"]=='variable','Expression Name']=18
    a=0
    for s in test.iloc[:,0]:
        test.iloc[a,0] = bytearray(s, 'utf-8')
        a+=1
    test_X_test = test.iloc[:,0]
    test_y_test = test.iloc[:,1]
    X_test = sequence.pad_sequences(test_X_test, maxlen=200)
    Y_test= np_utils.to_categorical(test_y_test)
    pre = model.predict(X_test)
    
    pre_list=[]
    for i in range(len(pre)) :
        if np.argmax(pre[i]) == 0 :
            pre_list.append('distributor_expression')
        elif np.argmax(pre[i]) == 1 :
            pre_list.append('linear_expression')
        elif np.argmax(pre[i]) == 2 :
            pre_list.append('linear_equation')
        elif np.argmax(pre[i]) == 3 :
            pre_list.append('inequality_equation')
        elif np.argmax(pre[i]) == 4 :
            pre_list.append('frac_multiplication_expression')
        elif np.argmax(pre[i]) == 5 :
            pre_list.append('frac_division_expression')
        elif np.argmax(pre[i]) == 6 :
            pre_list.append('exponential_expression_1')
        elif np.argmax(pre[i]) == 7 :
            pre_list.append('exponential_expression_1_1')
        elif np.argmax(pre[i]) == 8 :
            pre_list.append('exponential_expression_2')
        elif np.argmax(pre[i]) == 9 :
            pre_list.append('exponential_expression_3')
        elif np.argmax(pre[i]) == 10 :
            pre_list.append('exponential_expression_4')
        elif np.argmax(pre[i]) == 11 :
            pre_list.append('frac_exponential_expression_1')
        elif np.argmax(pre[i]) == 12 :
            pre_list.append('frac_exponential_expression_2')
        elif np.argmax(pre[i]) == 13 :
            pre_list.append('frac_exponential_expression_3')
        elif np.argmax(pre[i]) == 14 :
            pre_list.append('frac_exponential_expression_4')
        elif np.argmax(pre[i]) == 15 :
            pre_list.append('variable_expression1')
        elif np.argmax(pre[i]) == 16 :
            pre_list.append('variable_expression2')
        elif np.argmax(pre[i]) == 17 :
            pre_list.append('solution_equation')
        elif np.argmax(pre[i]) == 18 :
            pre_list.append('variable')
    
    #테스트 데이터에 예측값을 추가하여 DataFrame 형식으로 만들어준다.
    acc = []
    for i in range(len(pre)) :
        if pre_list[i] == test_target[i] :
            acc.append('O')
        else :
            acc.append('X')
    result = pd.DataFrame(data={'원본수식' : test_ME,'정답' : test_target, '예측값' : pre_list, '일치여부' : acc},
                          columns=['원본수식',"예측값",'정답','일치여부'],index=None)
    result=result.reset_index(drop=True)

    #예측이 틀린 수식에 대한 원본데이터 값과 예측값을 출력해준다.
    error_index=[]
    if acc.count('X')>0 :
        for i in range(len(acc)):
            if acc[i] == "X":
                error_index.append(i)
    for i in error_index :
        print("index : "+str(i))
        print("원본수식 :" + result.iloc[i,0]+ '\n'+"예측값 : " + result.iloc[i,1] + '\n'+"정답 : " + result.iloc[i,2]+"\n")
    print()
    print("정확도 : {}%, 테스트 {}문항 중 {}문항 예측실패".
          format(((len(pre)-len(error_index))/len(pre))*100,len(pre),len(error_index)))
    
    return result

        
def interval_model_fit(data_name,interval,epoch) :
    """
    
    새로운 예측 모델을 생성하고 싶을 때 데이터와 학습횟수의 최소 필요값을 유추하는데 도움을 주는 기능이다.
    새로운 예측모델 생성의 자동화라고도 생각할 수 있다.
    예를 들어 구분해야하는 수식의 갯수가 10개이고 데이터가 100만개 있다고 가정했을 때
    한번에 바로 100만개의 데이터를 학습시키는 것이 아니라 누적증분값을 설정한 후
    차례차례 증분값만큼 구간별로 데이터의 양을 늘려가며(ex. 1만개,2만개,3만개...) 모델이 학습을 진행하고 구간별 베스트 모델을 자동으로 저장해준다.
    모든 구간별 학습이 종료되면 베스트 모델에 대한 epoch값과 교차검증값을 그래프로 시각화하여 보여준다.
    
    :param data_name:학습에 사용할 데이터의 파일명(수식과 수식명이 csv 형식으로 이루어진 파일이어야 함)
    :param interval:누적증분값
    :param epoch:구간별 모델의 최대 학습횟수
    :return result:구간별 베스트 모델에 대한 정보가 담긴 DataFrame
    
    """
    startTime = time.time()
    
    data = pd.read_csv("{}".format(data_name),header=None)
    test_cnt=len(data.iloc[:,0])/interval
    data.columns=['Math Expresion','Expression Name']
    data.loc[data["Expression Name"]=='distributor_expression','Expression Name']=0
    data.loc[data["Expression Name"]=='linear_expression','Expression Name']=1
    data.loc[data["Expression Name"]=='linear_equation','Expression Name']=2
    data.loc[data["Expression Name"]=='inequality_equation','Expression Name']=3
    data.loc[data["Expression Name"]=='frac_multiplication_expression','Expression Name']=4
    data.loc[data["Expression Name"]=='frac_division_expression','Expression Name']=5
    data.loc[data["Expression Name"]=='exponential_expression_1','Expression Name']=6
    data.loc[data["Expression Name"]=='exponential_expression_1_1','Expression Name']=7
    data.loc[data["Expression Name"]=='exponential_expression_2','Expression Name']=8
    data.loc[data["Expression Name"]=='exponential_expression_3','Expression Name']=9
    data.loc[data["Expression Name"]=='exponential_expression_4','Expression Name']=10
    data.loc[data["Expression Name"]=='frac_exponential_expression_1','Expression Name']=11
    data.loc[data["Expression Name"]=='frac_exponential_expression_2','Expression Name']=12
    data.loc[data["Expression Name"]=='frac_exponential_expression_3','Expression Name']=13
    data.loc[data["Expression Name"]=='frac_exponential_expression_4','Expression Name']=14
    data.loc[data["Expression Name"]=='variable_expression1','Expression Name']=15
    data.loc[data["Expression Name"]=='variable_expression2','Expression Name']=16
    data.loc[data["Expression Name"]=='solution_equation','Expression Name']=17
    data.loc[data["Expression Name"]=='variable','Expression Name']=18
    
    cnt=0
    for s in data.iloc[:,0]:
        data.iloc[cnt,0] = bytearray(s, 'utf-8')
        cnt+=1
        
    start = 0
    i_cnt = interval
    earlystop = EarlyStopping(monitor='val_loss', mode='min',min_delta=0.0001, verbose = 1, patience=5)
    
    #변수 초기화 선언
    epoch_cnt=[]
    best_loss=[]
    data_cnt=[]
    best_score=[]
    
    #누적증분값에 따른 구간별 모델 학습 
    while(test_cnt>0):
        
        start += 1
        print("   {}번 구간 시작".format(start))
        print()
        mc = ModelCheckpoint('./models/interval_best_model_{}.h5'.format(i_cnt), monitor='val_loss', mode='min', save_best_only=True)
        data_i = data.iloc[:i_cnt,].copy()            
        X_train,X_test,y_train,y_test = train_test_split(data_i.iloc[:,0],data_i.iloc[:,1],test_size=0.2)
        X_train_seq = sequence.pad_sequences(X_train, maxlen=200)
        X_test_seq = sequence.pad_sequences(X_test, maxlen=200)
        Y_train_ctg = np_utils.to_categorical(y_train)
        Y_test_ctg = np_utils.to_categorical(y_test)
        model = Sequential()
        model.add(Embedding(len(X_train), 100))
        model.add(LSTM(100, activation='tanh'))
        model.add(Dense(19, activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        history = model.fit(X_train_seq, Y_train_ctg, batch_size=128, 
                        epochs=epoch,validation_data=(X_test_seq, Y_test_ctg),callbacks=[earlystop,mc])
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
    result=result.set_index('학습데이터량')
    
    endTime = time.time() - startTime
    print('총 소요시간 : %.2f 초' % endTime) 
    
    #결과값 그래프로 시각화
    plt.rcParams["figure.figsize"] = (15,7)
    fig, loss_ax = plt.subplots()
    epoch_ax = loss_ax.twinx()
    loss_ax.plot(result['Epoch'], 'g', label='Epoch',marker='o')
    epoch_ax.plot(result['손실최소값'], 'b', label='best val_loss',marker='o')
    loss_ax.set_xlabel('data',fontsize=18)
    loss_ax.set_ylabel('Epoch',fontsize=18)
    epoch_ax.set_ylabel('Val_loss',fontsize=18)
    loss_ax.legend(loc='lower left',fontsize=15)
    epoch_ax.legend(loc='upper left',fontsize=15)

    plt.show()
    
    return result