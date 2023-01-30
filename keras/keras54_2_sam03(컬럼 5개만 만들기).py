# 이메일 제목 고정: 한태희 00,000원
#첨부파일 소스, 가중치
# 삼성전자 월요일 시가

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten, LSTM
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. data 전처리
samdf01_csv=pd.read_csv('C:/study/_data/sam01/삼성전자 주가.csv',index_col=[0],encoding='cp949',thousands=',').loc[::-1] # thousands를 통해 object문제 해결 -> int64, float64 같이 써도 문제 없는가? -> 문제 없음 오히려 int64를 float64로 바꾸면 소수점 데이터 손실+데이터 용량 증가 발생.
amodf01_csv=pd.read_csv('C:/study/_data/sam01/아모레퍼시픽 주가.csv',index_col=[0],sep=',',encoding='cp949', thousands=',').loc[::-1]
print(samdf01_csv.info()) 
print(amodf01_csv.info()) 

samdf01_csv=samdf01_csv.dropna() # 결측치 있는 행 삭제
amodf01_csv=amodf01_csv.dropna() # 결측치 있는 행 삭제




# amodf01_csv = samdf01_csv.sample(n=len(samdf01_csv))
amodf01_csv = amodf01_csv.loc[1986:0,['시가', '고가', '저가', '종가', '거래량']]
print(amodf01_csv.shape)
amodf01_csv.tail()


x1=samdf01_csv[['시가', '고가', '저가', '종가', '거래량']]
x2=amodf01_csv[['시가', '고가', '저가', '종가', '거래량']]
y=samdf01_csv['시가']
y.tail()
print(x1.shape) # (1977, 5)
print(x2.shape) # (1977, 5)
print(y.shape) # (1977,)
amodf01_csv.tail()
x1.tail()

def split_xy(dataset, timesteps,y_column): # timesteps = 특정 길이,열
    x,y=list(),list()
    for i in range(len(dataset)): # i = 인덱스 (행) 
        x_end_number = i+timesteps # x의 끝자리 = 해당 인덱스 번호 + 필요한 나머지 컬럼개수만큼[[a][b][c][d]]
        y_end_number = x_end_number+y_column
        if y_end_number > len(dataset):
            break
        tmp_x=dataset[i:x_end_number,:] #  i번째부터 x_end_number번째까지의, 모든 열에서의 데이터를 가져옴 -> tep_x 저장
        tmp_y=dataset[x_end_number:y_end_number,1] # x_end_number부터 y_end_number까지의 1번째 인덱스(시가) 열의 데이터만 가져옴 -> tep_y 저장
        x.append(tmp_x) # tmp_x라는 변수에 저장된 값을 x 리스트에 추가
        y.append(tmp_y) # tmp_y라는 변수에 저장된 값을 y 리스트에 추가
    return np.array(x), np.array(y)
x1,y1=split_xy(samdf01_csv,5,1)
x2,y2=split_xy(amodf01_csv,5,1)
print(x1[0,:],"\n", y1[0]) # 모든 열에서의 첫행 x1 data 출력 / 첫행 y1 data 출력
print(x2[0,:],"\n", y2[0]) # 모든 열에서의 첫행 x2 data 출력 / 첫행 y2 data 출력
print(x2.shape) # (2198, 12, 12)
print(y2.shape) # (2198, 1)

#1.8) test data, train data 분리
from sklearn.model_selection import train_test_split

x1_train,x1_test,x2_train,x2_test,y_train,y_test=train_test_split(
    x1,x2,y,
    train_size=0.8,
    random_state=1234,
)

#1.9) data 분류 후 train, test data 양 확인
print("______________")
print(x1_train.shape) # (1581, 5)
print(x1_test.shape) # (396, 5)
print(x2_train.shape) # (1581, 5)
print(x2_test.shape) # (396, 5)
print(y_train.shape) # (1581,)
print(y_test.shape) # (396,)


"""
# #1.6) pandas numpy 변환
# samdf01_csv=samdf01_csv.values
# print(type(samdf01_csv))
# amodf01_csv=amodf01_csv.values
# print(type(amodf01_csv))

# #1.7) x, y 나누기

# def split_xy(dataset, timesteps,y_column): # timesteps = 특정 길이,열
#     x,y=list(),list()
#     for i in range(len(dataset)): # i = 인덱스 (행) 
#         x_end_number = i+timesteps # x의 끝자리 = 해당 인덱스 번호 + 필요한 나머지 컬럼개수만큼[[a][b][c][d]]
#         y_end_number = x_end_number+y_column
#         if y_end_number > len(dataset):
#             break
#         tmp_x=dataset[i:x_end_number,:] #  i번째부터 x_end_number번째까지의, 모든 열에서의 데이터를 가져옴 -> tep_x 저장
#         tmp_y=dataset[x_end_number:y_end_number,1] # x_end_number부터 y_end_number까지의 1번째 인덱스(시가) 열의 데이터만 가져옴 -> tep_y 저장
#         x.append(tmp_x) # tmp_x라는 변수에 저장된 값을 x 리스트에 추가
#         y.append(tmp_y) # tmp_y라는 변수에 저장된 값을 y 리스트에 추가
#     return np.array(x), np.array(y)
# x1,y1=split_xy(samdf01_csv,12,1)
# x2,y2=split_xy(amodf01_csv,12,1)
# print(x1[0,:],"\n", y1[0]) # 모든 열에서의 첫행 x1 data 출력 / 첫행 y1 data 출력
# print(x2[0,:],"\n", y2[0]) # 모든 열에서의 첫행 x2 data 출력 / 첫행 y2 data 출력
# print(x2.shape) # (2198, 12, 12)
# print(y2.shape) # (2198, 1)

"""

# model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1 모델1
input1=Input(shape=(5,))
dense1=Dense(11,activation='relu', name='ds11')(input1)
dense2=Dense(12,activation='relu', name='ds12')(dense1)
dense3=Dense(13,activation='relu', name='ds13')(dense2)
output1=Dense(14,activation='relu', name='ds14')(dense3)


#2-2 모델2
input2=Input(shape=(5,))
dense21=Dense(21,activation='relu', name='ds21')(input2)
dense22=Dense(22,activation='relu', name='ds22')(dense21)
output2=Dense(23,activation='relu', name='ds23')(dense22)


#2-3 모델 병합
from tensorflow.keras.layers import concatenate
merge1=concatenate([output1,output2],name='mg1')
merge2=Dense(15, activation='relu',name='mg2')(merge1)
merge3=Dense(15,name='mg3')(merge2)
last_output=Dense(1,name='last')(merge3)

model=Model(inputs=[input1,input2],outputs=last_output)

model.summary()


#3. compile, training
model.compile(
    loss='mae',
    optimizer='adam',
    metrics=['mse']
)


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=2,
    restore_best_weights=True
)


import datetime
date=datetime.datetime.now()

date=date.strftime("%m%d_%H%M")

model_checkpoint=ModelCheckpoint(
    filepath='c:/study/_save/MCP/' + 'k54_sam01_' + date + '_{epoch}-{val_loss:.4f}.h5',
    monitor='val_loss',
    verbose=2,
    save_best_only=True
)


hist=model.fit(
    [x1_train,x2_train],y_train,
    epochs=75752576,
    batch_size=1,
    verbose=2,
    validation_split=0.2,
    callbacks=[early_stopping]
)

#4. 평가, 예측
loss=model.evaluate([x1_test,x2_test],y_test)



y_predict = model.predict([x1_test,x2_test])

for i in range(20):
    print('시가 : ', y_test[i], '/ 예측가 : ', y_predict[i])
print('loss : ',loss)

"""
종가 :  41400 / 예측가 :  [41803.41]
종가 :  81700 / 예측가 :  [81662.5]
종가 :  1355000 / 예측가 :  [1363921.4]
종가 :  1321000 / 예측가 :  [1330712.8]
종가 :  1067000 / 예측가 :  [1046870.]
"""

