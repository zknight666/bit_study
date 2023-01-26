#cnn = 4차원, dnn=2차원 이상 , input_shape= x값 차원 -1, rnn = 3차원
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN
from tensorflow.keras.models import Sequential




#1. data
dataset= np.array([1,2,3,4,5,6,7,8,9,10]) # (10,)

x=np.array([
    [1,2,3],
    [2,3,4],
    [3,4,5],
    [4,5,6],
    [5,6,7],
    [6,7,8],
    [7,8,9]
    ]) # 3개씩 나눔, ★ RNN 모델은 data를 직접 나눠줘야함 -> ex) https://minibig.tistory.com/24
y=np.array([4,5,6,7,8,9,10])
# 1,2,3, => 4  / 2,3,4 => 5 .....   7,8,9 => 10

print(x.shape) #(7,3)
print(y.shape) # (7,)


x=x.reshape(7,3,1) # ★ RNN모델 reshape 필요 ->  [[[1],[2],[3]],[[2],[3],[4]], ...]

print(x.shape) # (7, 3, 1) # ★★ 총 7개의 batch를 3개씩 묶은 것을 1개씩 연산(훈련)했다.



"""
1,2,3 | 4
2,3,4 | 5
3,4,5 | 6
4,5,6 | 7
5,6,7 | 8
6,7,8 | 9
7,8,9 | 10
# 7개의 ??을 3개씩 자른 것을 1개씩 연산했다.
"""


#2. model
model=Sequential()
model.add(SimpleRNN(254,input_shape=(3,1),activation='relu'))
# model.add(SimpleRNN(units=254,input_shape=(batch, timesteps, feature),activation='relu'))
# input_shape = 3차원 텐서 3d tensor = (행,묶음, 연산 횟수)
model.add(Dense(500,activation='relu'))
model.add(Dense(1))
# ★ RNN input_shape 조심할 것, 순환 모델 형식임을 인식
model.summary()

"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 simple_rnn_2 (SimpleRNN)    (None, 254)               65024

 dense_7 (Dense)             (None, 500)               127500

 dense_8 (Dense)             (None, 1)                 501

=================================================================
Total params: 193,025
Trainable params: 193,025
Non-trainable params: 0

rnn 64
dense 16

DNN 인 경우
64*4
256개

RNN인 경우
64 * (64+1+1) = 4224
units * (feature + bias + units) = parms


"""



#3. compile, training
model.compile(optimizer='adam',loss='mse',metrics='mae')
model.fit(x,y,batch_size=1,epochs=500)


#4. 평가, 예측
results=model.evaluate(x,y)

y_predict=np.array([8,9,10]).reshape(1,3,1) # (nan,3,1) = (1,3,1), # rehsape 필요
result=model.predict(y_predict)
print('8,9,10 결과 : ',result)

print('mse : ',results[0])
print('mae : ',results[1])


