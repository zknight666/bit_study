import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
# 1개의 입력, 출력을 가진 모델, layer를 차례로 쌓는 모델
from tensorflow.keras.layers import Dense
# dense = 입력 출력 연결 layer

#1. data
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

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
model.add(Dense(412))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(99))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

"""
layer 개수, node의 개수를 높이는 것으로 성능 향상 기대 가능
hyperparameter tuning
튜닝 가능한 파라미터 : batch_size, epochs, optimizers, learning rate, activation, Regularization(weight decay, Dropout, 은닉층(Hidden layer)의 노드(Node) 수
적절한 가중치 저장
1) batch_size
  : 데이터 나눠서 저장하도록 하는 변수, batch size를 낮게할수록 시간이 오래걸리므로 적절한 값을 찾아야함
  1)) 데이터 양 많을때 조절하는 변수(과적합 방지), 2)) 메모리 조절
    batch_size=1 -> batch 작업 6번 / batch_size=4 -> batch 작업 2번, 4개 1번, 나머지 1번 / batch size default value = 32
loss 최적화 변수 : epochs, layer, node
"""
#3. compile & training
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=10, batch_size=2)



#4. 평가, 예측

loss = model.evaluate(x,y)
print('loss : ', loss)
"""
predict는 좋은데 loss값이 잘 안나온 경우
    판단의 기준은 loss로 봐야함.
"""

result = model.predict([7])
print('6의 결과 : ', result)


