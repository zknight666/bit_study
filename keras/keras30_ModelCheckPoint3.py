from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. data (data 불러오기, 컬럼 확인, 전처리)

#data 불러오기
dataset=load_boston()

x=dataset.data
y=dataset.target


# column 확인
print(x.shape) # (506, 13)
print(y.shape) # (506,)


# data 전처리 (1)
x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    train_size=0.8,
    random_state=333,
    shuffle=True
)

# data 전처리 (2)
scaler_minmax=MinMaxScaler()# train data 0 ~ 0.8만 훈련 data 기준 scaling x가지고 scaling하면 0~1 y_predict할때는 이외의 값이 튀어나올 수 있음 
x_train=scaler_minmax.fit_transform(x_train)
x_test=scaler_minmax.transform(x_test) # test에는 fit 쓰면 안됨
print(np.min(x)) # MinMaxScaler 적용됬는지 np.min,max로 확인 최소값 0
print(np.max(x)) # 최대값 1로 min,max 제대로 적용 확인 완료



# 2-2 model (함수형) # Model import 필요, input layer 명시해주어야함 -> Input import 필요
input1=Input(shape=(13,))
dense1=Dense(50,activation='relu')(input1)
dense2=Dense(40)(dense1)
dense3=Dense(30)(dense2)
dense4=Dense(20)(dense3)
dense5=Dense(10)(dense4)
output1=Dense(1,activation='relu')(dense5)
model=Model(inputs=input1,outputs=output1)

model.summary() # 함수형 순차형 동일한 모델 params 같음

#3. compile, training
model.compile(
    optimizer='nadam',
    loss='mae',
    metrics=['mse']
)


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=30,
    verbose=2,
    restore_best_weights=True
)

model_checkpoint=ModelCheckpoint(
    filepath='c:/study/_save/MCP/keras30_ModelCheckPoint3.hdf5', 
    monitor='val_loss',
    verbose=2,
    save_best_only=True,
    # save_freq=2
)
#확장자 h5, hdf5 같은 확장자 



hist=model.fit(
    x_train,y_train,
    epochs=1000,
    batch_size=1,
    verbose=2, # 일반적으로 1=모두 보여줌, 0 = 화면 안나옴,2= 생략해서 보여줌, 나머지=epoch 횟수만 보여줌 verbose 0으로 두면 계산속도가 더 많이 빨라짐.
    # verbose 1= 13초 / verbose=0 = 10초
    callbacks=[early_stopping, model_checkpoint],
    validation_split=0.2
)

model.save('c:/study/_save/keras30_save_model.hdf5')



# model=load_model(
#     filepath='c:/study/_save/MCP/keras30_ModelCheckPoint1.hdf5')


#4. 평가, 예측
mse,mae=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
def RMSE(y_test,y_predict):
    return(np.sqrt(mean_squared_error(y_test,y_predict)))



print('===========================1. 기본 출력=========================')

print('mse: ',mse)
print('mae: ',mae)
print('RMSE : ',RMSE(y_test,y_predict))
print('r2: ',r2_score(y_test,y_predict))

print('===========================2. load model 출력=========================')
model2=load_model('c:/study/_save/keras30_save_model.hdf5')
y_predict=model2.predict(x_test)
mse,mae=model2.evaluate(x_test,y_test)

print('mse: ',mse)
print('mae: ',mae)
print('RMSE : ',RMSE(y_test,y_predict))
print('r2: ',r2_score(y_test,y_predict))

print('===========================3. modelcheckpoint 출력=========================') # load model이랑 별 차이 없음
model3=load_model('c:/study/_save/MCP/keras30_ModelCheckPoint3.hdf5')
y_predict=model3.predict(x_test)
mse,mae=model3.evaluate(x_test,y_test)

print('mse: ',mse)
print('mae: ',mae)
print('RMSE : ',RMSE(y_test,y_predict))
print('r2: ',r2_score(y_test,y_predict))






# {'키(loss)':[value1,value2,3..])}
# 데이터 형태 = 리스트 => ['ㅁ','ㅠ','ㅊ'], 딕셔너리 => {'ㅁ':0,'ㅠ':2,'ㅊ':5} (딕셔너리는 {'키':value} 형태)





"""
결과
scaler 없었을 때
loss :  [2.902878761291504, 0.0]
RMSE :  4.0858269073324704
r2 :  0.8310794041312348

scaler 썼을 때 # 좋지 않은 경우도 존재
loss:  [2.4661803245544434, 14.00954818725586]
RMSE :  3.742933349597218
r2:  0.8571605108056923

standard scaler 썼을 때 # 
loss:  [2.4749691486358643, 19.448814392089844]
RMSE :  4.410080274362113
r2:  0.8017025677403397
"""

# mcp 저장
# loss:  [2.4320850372314453, 16.755176544189453]
# RMSE :  4.0933087953318115
# r2:  0.829166468678755

# MCP 불러오기
# loss:  [2.4320850372314453, 16.755176544189453]
# RMSE :  4.0933087953318115
# r2:  0.829166468678755