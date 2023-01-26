import numpy as np
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN
from tensorflow.keras.models import Sequential


#1. data

x=np.array([
    [1,2,3],
    [2,3,4],
    [3,4,5],
    [4,5,6],
    [5,6,7],
    [6,7,8],
    [7,8,9],
    [8,9,10],
    [9,10,11],
    [10,11,12],
    [20,30,40],
    [30,40,50],
    [40,50,60]
    ]) # 3개씩 나눔, ★ RNN 모델은 data를 직접 나눠줘야함 -> ex) https://minibig.tistory.com/24

y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape) #(13, 3)
print(y.shape) # (13,)




x=x.reshape(13,3,1) # ★ RNN모델 reshape 필요 ->  [[[1],[2],[3]],[[2],[3],[4]], ...]

print(x.shape) # (13, 3, 1) # ★★ 총 7개의 batch를 3개씩 묶은 것을 1개씩 연산(훈련)했다.


#2. model
model=Sequential()
model.add(LSTM(units=10,input_shape=(3,1), activation='relu',return_sequences=True))
model.add(LSTM(10))
#input = (n,3,1) -> 출력 -> (n,10) 로 출력 -> LSTM은 3차원 받아야함 -> 3차원 변환 필요 -> return_sequences=True -> LSTM 반복 가능
# model.add(LSTM(10))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))
model.summary()



#3. compile, training
model.compile(optimizer='adam',loss='mse',metrics='mae')
model.fit(x,y,batch_size=1,epochs=50)



#4. 평가, 예측
results=model.evaluate(x,y)
print('mse : ',results[0])
print('mae : ',results[1])

x_predict=np.array([50,60,70]).reshape(1,3,1) # (nan,3,1) = (1,3,1), # rehsape 필요
result=model.predict(x_predict)
print('50,60,70 결과 : ',result)

# 50,60, 70 을 넣어서 80 나오게 하기




# DNN








