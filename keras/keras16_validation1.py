import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pandas as pd

#1. data 처리 (불러오기, 전처리, )

x_train = np.array(range(1,11))
y_train = np.array(range(1,11))
x_test = np.array([11,12,13])
y_test = np.array([11,12,13])
x_validation = np.array([14,15,16])
y_validation = np.array([14,15,16])







#2. 모델 구성
model=Sequential()
model.add(Dense(1,input_dim=1))
model.add(Dense(3,activation='relu'))
model.add(Dense(1))




# 컴파일, 훈련


model.compile(
    loss='mse',
    optimizer='adam',
    )

model.fit(
    x_train,y_train,
    epochs=100,
    validation_data=(x_validation,y_validation)
)



#4. 평가, 예측

loss=model.evaluate(x_test,y_test)
print('loss : ',loss)

result=model.predict([17])
print('17 예측 값 : ',result)





