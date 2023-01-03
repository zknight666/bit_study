import numpy as np
import tensorflow as tf
print(tf.__version__) # 2.7.4

#1. 정제된 data / data 전처리 *이게 핵심
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])

# shift + enter = 소스코드 한줄씩 실행 가능

#2.model 껍데기, 명령어 y = wx + b
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1))


#3.compile & training
model.compile(loss='mae', optimizer='adam')

model.fit(x, y, epochs=200)
#batch = 묶음 처리 / iteration = 반복
# 

#4.평가 & 예측
results = model.predict([6])
print('6의 예측값 : ', results)

