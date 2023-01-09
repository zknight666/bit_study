from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import time





# 함수 클래스 차이
# 


#1. data (data 불러오기, 컬럼 확인, 전처리)

#data 불러오기
dataset=load_boston()

x=dataset.data
y=dataset.target

# column 확인
print(x.shape) # (506, 13)
print(y.shape) # (506,)



# print(dataset.describe()) 왜 안돠

# data 전처리
x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    train_size=0.8,
    random_state=333,
    shuffle=True
)




#2. model
model=Sequential()
# model.add(Dense(1,input_dim=13)) # (?,13)
model.add(Dense(5,input_shape=(13,))) # (13,?)
model.add(Dense(500,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(5,activation='relu'))
# model.add(Dense(1,activation='relu'))
# model.add(Dense(1,activation='relu'))
model.add(Dense(1))





#3. compile, training
model.compile(
    optimizer='adam',
    loss='mae',
    metrics=['mse']
)

start=time.time()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=2,
    restore_best_weights=True
)


model.fit(
    x_train,y_train,
    epochs=1000,
    batch_size=32,
    verbose=2, # 일반적으로 1=모두 보여줌, 0 = 화면 안나옴,2= 생략해서 보여줌, 나머지=epoch 횟수만 보여줌 verbose 0으로 두면 계산속도가 더 많이 빨라짐.
    # verbose 1= 13초 / verbose=0 = 10초
    callbacks=[early_stopping],
    validation_split=0.2
)


end=time.time()




#4. 평가, 예측

loss=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)

def RMSE(y_test,y_predict):
    return(np.sqrt(mean_squared_error(y_test,y_predict)))





#5. 파일 만들기 ()





print('loss: ',loss)
print('걸린 시간 : ',end-start)
print('RMSE : ',RMSE(y_test,y_predict))
print('r2: ',r2_score(y_test,y_predict))


