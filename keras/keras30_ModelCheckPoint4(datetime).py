from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import datetime

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
    random_state=1234,
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






#######파일 이름 자동으로 만드는 방법######
date=datetime.datetime.now() # 현재 시간 표시 # windows 시간 기준이라 시간 틀린 경우도 그대로 입력됨.
print(date) # 2023-01-12 14:57:49.570040
print(type(date)) # <class 'datetime.datetime'> datetime이라는 타입임. 파일 명에다 집어 넣으려면 문자형으로 바뀌어야함.
date=date.strftime("%m%d_%H%M") # 날짜타입을 문자타입으로 바꾸는 명령어 %m=달, %d=일 _ %H=시간 %M=분
print(date) # 0112_1502
print(type(date)) # <class 'str'>

filepath = 'c:/study/_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # 04d = 정수 4자리까지 나와라, .4f = 소수 4째 자리까지 나와라 # 실제 나타나는 형태 : #0037-0.0048.hdf5
## 가운데 - 는 문자열, 중괄호 {}는 안에 있는 내용물을 (epoch,val_loss를) 땡겨온다라는 의미



model_checkpoint=ModelCheckpoint(
    filepath= filepath + 'k30_' + date + '_' + filename, 
    monitor='val_loss',
    verbose=2,
    save_best_only=True,
)
# 최종 파일 명 : k30_0112_1527_0038-2.6779.hdf5 1월12일 3시 27분_38번째 epochs,val_loss=2.678 
#######################################################################################





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




# {'키(loss)':[value1,value2,3..])}
# 데이터 형태 = 리스트 => ['ㅁ','ㅠ','ㅊ'], 딕셔너리 => {'ㅁ':0,'ㅠ':2,'ㅊ':5} (딕셔너리는 {'키':value} 형태)