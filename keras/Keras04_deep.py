import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
# 1개의 입력, 출력을 가진 모델, layer를 차례로 쌓는 모델
from tensorflow.keras.layers import Dense
# dense = 입력 출력 연결 layer

#1. data
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])

#2. model
model = Sequential()
#sequential 명령어를 model이라는 이름으로 정의하겠다.
model.add(Dense(3, input_dim=1))

#1 layer -> 3layer
model.add(Dense(5))
#3 layer -> 5layer, hidden layer는 input dim 안적어도 됨
model.add(Dense(4))
#5 layer -> 4layer
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
#layer 개수, node의 개수를 높이는 것으로 성능 향상 기대 가능
#hyperparameter tuning
#튜닝 가능한 파라미터 : batch_size, epochs, optimizers, learning rate, activation, Regularization(weight decay, Dropout, 은닉층(Hidden layer)의 노드(Node) 수

#3. compile & training
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=200, batch_size=3)



#4. 평가
result = model.predict([6])
print('6의 결과 : ', result)


