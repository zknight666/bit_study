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
samdf01_csv=pd.read_csv('C:/study/_data/sam01/삼성전자 주가.csv',index_col=[0],sep=',',encoding='cp949',thousands=',').loc[::-1] # thousands를 통해 object문제 해결 -> int64, float64 같이 써도 문제 없는가? -> 문제 없음 오히려 int64를 float64로 바꾸면 소수점 데이터 손실+데이터 용량 증가 발생.
amodf01_csv=pd.read_csv('C:/study/_data/sam01/아모레퍼시픽 주가.csv',index_col=[0],sep=',',encoding='cp949', thousands=',').loc[::-1]
print(samdf01_csv.info())
print(amodf01_csv.info())
# samdf01_csv=samdf01_csv.dropna() # 결측치 있는 행 삭제
# amodf01_csv=amodf01_csv.dropna() # 결측치 있는 행 삭제

samdf02=samdf01_csv[['시가', '고가', '저가', '종가', '기관']]
amodf02=amodf01_csv[['시가', '고가', '저가', '종가', '기관']]
sam_y=samdf01_csv['시가']
sam_y=sam_y.values
print(samdf02.info())
print(amodf02.info())
print(np.info(sam_y))

amodf02 = samdf02.sample(n=len(samdf02)) # 날짜 랜덤으로 바꿔줌 날짜 오름차순으로 변경해야할듯
amodf02 = amodf02.sort_values(by='일자', ascending=True)
amodf02.tail()
print(samdf02.info())
print(amodf02.info())

# TypeError: '(slice(0, 5, None), slice(None, None, None))' is an invalid key
# -> 데이터 집합 인수가 팬더 데이터 프레임이지만 코드가 numpy 스타일 슬라이싱을 사용하여 요소에 액세스하려고 하기 때문에 발생합니다.
amodf02=amodf02.values
samdf02=samdf02.values
print(type(amodf02)) # <class 'pandas.core.frame.DataFrame'>
print(type(samdf02)) # <class 'pandas.core.frame.DataFrame'>


np.save('C:/study/_data/sam01/samdf02.npy',arr=samdf02)
np.save('C:/study/_data/sam01/amodf02.npy',arr=amodf02)

def split_xy(dataset, timesteps,y_column): # timesteps = 내가 필요한 날짜까지, +행
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
x1,y1=split_xy(samdf02,10,1) # ★★ 3차원으로 변환
x2,y2=split_xy(amodf02,10,1)
print(x1[0,:],"\n", y1[0]) # 모든 열에서의 첫행 x1 data 출력 / 첫행 y1 data 출력
print(x2[0,:],"\n", y2[0]) # 모든 열에서의 첫행 x2 data 출력 / 첫행 y2 data 출력
print(x1.shape) # (1972, 5, 5)
print(x2.shape) # (1972, 5, 5)
print(y1.shape) # (1972, 1)
print(y2.shape) # (1972, 1)


#1.8) test data, train data 분리
from sklearn.model_selection import train_test_split

x1_train,x1_test,x2_train,x2_test,y1_train,y1_test,y2_train,y2_test=train_test_split(
    x1,x2,y1,y2,
    train_size=0.8,
    random_state=1234,
    shuffle=False
)

#1.9) data 분류 후 train, test data 양 확인
print("______________")
print(x1_train.shape) # (1577, 5, 5)
print(x1_test.shape) # (395, 5, 5)
print(x2_train.shape) # (1581, 5)
print(x2_test.shape) # (395, 5, 5)
print(y1_train.shape) # (1577, 1)
print(y1_test.shape) # (395, 1)
print(y2_train.shape) # (1577, 1)
print(y2_test.shape) #(395, 1)

x1_train = np.reshape(x1_train,(x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2])) # (row,column,차원크기) # ★★  2차원으로 변환
x1_test = np.reshape(x1_test,(x1_test.shape[0], x1_test.shape[1] * x1_test.shape[2]))
x2_train = np.reshape(x2_train,(x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2]))
x2_test = np.reshape(x2_test,(x2_test.shape[0], x2_test.shape[1] * x2_test.shape[2]))
print(x2_train.shape) # (1577, 25)
print(x2_test.shape) # (395, 25)

from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
scaler1.fit(x1_train)
x1_train_scaled = scaler1.transform(x1_train)
x1_test_scaled = scaler1.transform(x1_test)
scaler2 = StandardScaler()
scaler2.fit(x2_train)
x2_train_scaled = scaler2.transform(x2_train)
x2_test_scaled = scaler2.transform(x2_test)
print(x2_train_scaled[0, :])

x1_train_scaled = np.reshape(x1_train_scaled,(x1_train_scaled.shape[0], 10,5))
x1_test_scaled = np.reshape(x1_test_scaled,(x1_test_scaled.shape[0], 10,5))
x2_train_scaled = np.reshape(x2_train_scaled,(x2_train_scaled.shape[0], 10,5))
x2_test_scaled = np.reshape(x2_test_scaled,(x2_test_scaled.shape[0], 10,5))
print(x2_train_scaled.shape) # (1577, 5, 5)
print(x2_test_scaled.shape) # (395, 5, 5)

# model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1 모델1
input1 = Input(shape=(10,5))
dense1 = LSTM(20)(input1)
dense1 = Dense(32)(dense1)
dense1 = Dense(32)(dense1)
output1 = Dense(32)(dense1)

input2 = Input(shape=(10, 5))
dense2 = LSTM(20)(input2)
dense2 = Dense(64)(dense2)
dense2 = Dense(64)(dense2)
dense2 = Dense(64)(dense2)
output2 = Dense(32)(dense2)

from keras.layers.merge import concatenate
merge1=concatenate([output1,output2])
merge2=Dense(12, activation='relu')(merge1)
merge3=Dense(13)(merge2)
last_output=Dense(1)(merge3)
model = Model(inputs=[input1, input2],outputs=last_output)
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


hist = model.fit(
    [x1_train_scaled, x2_train_scaled], y1_train, 
    validation_split=0.2, 
    verbose=2, 
    batch_size=1, 
    epochs=100, 
    callbacks=[early_stopping]
    )



#4. 평가, 예측

loss, mse = model.evaluate([x1_test_scaled, x2_test_scaled], y1_test, batch_size=1)

print('loss : ', loss)
print('mse : ', mse)

y1_pred = model.predict([x1_test_scaled, x2_test_scaled])


for i in range(10):
    print('시가 : ', y1_test[i], '/ 예측가 : ', y1_pred[i])
    

"""
종가 :  41400 / 예측가 :  [41803.41]
종가 :  81700 / 예측가 :  [81662.5]
종가 :  1355000 / 예측가 :  [1363921.4]
종가 :  1321000 / 예측가 :  [1330712.8]
종가 :  1067000 / 예측가 :  [1046870.]
"""

