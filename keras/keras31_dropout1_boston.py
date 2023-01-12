import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#1. data (data 불러오기, 컬럼 확인, 전처리)


#data 불러오기
dataset=load_boston()

x=dataset.data
y=dataset.target


# column 확인
print(x.shape) # (506, 13)
print(y.shape) # (506,)



# data 전처리
x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    train_size=0.8,
    random_state=333,
    shuffle=True
)


scaler_minmax=MinMaxScaler()
x_train=scaler_minmax.fit_transform(x_train)
x_test=scaler_minmax.transform(x_test) 
print(np.min(x))
print(np.max(x)) 


#2-2 model (함수형) # Model import 필요, input layer 명시해주어야함 -> Input import 필요
input1=Input(shape=(13,))
dense1=Dense(500,activation='relu')(input1)
dropout1=Dropout(rate=0.2)(dense1) # 훈련시킬때만 적용.test는 다 적용됨.
dense2=Dense(40)(dropout1) 
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
    patience=50,
    verbose=2,
    restore_best_weights=True
)




import datetime
date=datetime.datetime.now()
date=date.strftime("%m%d_%H%M")

model_checkpoint=ModelCheckpoint(
    filepath='c:/study/_save/MCP/' + 'K31_boston_' + date + '_{epoch}-{val_loss:.4f}.h5',
    monitor='val_loss',
    verbose=2,
    save_best_only=True
)


hist=model.fit(
    x_train,y_train,
    epochs=1000,
    batch_size=1,
    verbose=2,
    callbacks=[early_stopping, model_checkpoint],
    validation_split=0.2
)

import datetime





#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)

def RMSE(y_test,y_predict):
    return(np.sqrt(mean_squared_error(y_test,y_predict)))




# 결과 값

print('loss: ',loss)
print('RMSE : ',RMSE(y_test,y_predict))
print('r2: ',r2_score(y_test,y_predict))




#6. 시각화


plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue',marker='.', label='val_loss') # epoch 순으로 가서 x값 생략해도 됨
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('boston loss')
plt.legend(loc='upper right') # 'upper left'
plt.show()



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

dropout+minmax scaler 썼을 때
loss:  [2.4789412021636963, 14.94411849975586]
RMSE :  3.8657621564517832
r2:  0.8476317913959905
dropout+minmax scaler 썼을 때 (2)
mse:  2.339980125427246
mae:  11.263151168823242
RMSE :  3.356061831537637
r2:  0.8902749219511319

"""

