# 이미지 data가 아닌 것들을 CNN 모델로 돌려보기 ->  -> (404,13,1,1) => input_shape =(13,1,1) 가로 13개, 세로 1개, 흑백

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#1. data (data 불러오기, 컬럼 확인, 전처리)


#1) data 불러오기
dataset=load_boston()

x=dataset.data
y=dataset.target


#2) column 확인
print(x.shape) # (506, 13)
print(y.shape) # (506,)



#3) data 전처리
x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    train_size=0.8,
    random_state=333,
    shuffle=True
)

#4) 나눠진 train, test data 양 확인
print(x_train.shape) # (404, 13) -> (404,13,1,1) => input_shape =(13,1,1) 가로 13개, 세로 1개, 흑백
print(x_test.shape) # (102, 13)

#5) CNN모델을 위한 차원 변환
x_train=x_train.reshape(404,13,1,1)
x_test=x_test.reshape(102,13,1,1)

#6) 변환 확인
print(x_train.shape) # (404, 13, 1, 1)
print(x_test.shape) # (102, 13, 1, 1)


# 7) scaler 사용
# scaler_minmax=MinMaxScaler()
# x_train=scaler_minmax.fit_transform(x_train)
# x_test=scaler_minmax.transform(x_test)

# 8) 변환 확인
print(np.min(x))
print(np.max(x)) 



#2. model
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(2,1),input_shape=(13,1,1),activation='relu', padding='same'))
#커널 사이즈 조심할 것
model.add(Conv2D(filters=32,kernel_size=(2,1),activation='relu', padding='same')) # (27,27,128)
model.add(MaxPooling2D(pool_size=(2,1)))
model.add(Dropout(rate=0.3))
model.add(Dense(512,activation='relu')) #
model.add(Dropout(rate=0.3))
model.add(Flatten())
model.add(Dense(1,activation='relu'))


model.summary() # 함수형 순차형 동일한 모델 params 같음


#3. compile, training
model.compile(
    optimizer='adam',
    loss='mae',
    metrics=['mse']
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
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
    callbacks=[early_stopping],
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

#캘리포니아 = (n,8), = (n,2,2,2)= (n,4,2,1)=(n,8,1,1)
