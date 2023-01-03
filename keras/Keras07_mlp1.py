import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#mlp=multi layer perceptrone

#1. 데이터

x=np.array([[1,2,3,4,5,6,7,8,9,10], 
           [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
y=np.array([2,4,6,8,10,12,14,16,18,20])
# input_dim 2개, output_dim=1개
print(x.shape) #(2, 10) / 10개 데이터 2묶음 2행 10열

""" 
[1] = (1) 1개의 스칼라를 가진 벡터
[1,2] = (2, ) 2개의 스칼라를 가진 1개의 벡터(벡터부터 shape로 표현 가능)
[[1],[2],[3]] = (3,1) 1개짜리벡터가 3개
[[1,2,3]] = (1,3) 3개짜리벡터가 1개
[[1,2],[3,4]] = (2,2)
[[[1,2,3]]] = (1,1,3)
[[1,2,3],[4,5,6]] = (2,3) 
[[[1],[2]]] = (1,2,1)
[[1,2,3],[4,5]] = x
[[1,2],[1,2]] 
과제 행렬 10개 만들어서 이면지로 제출

 """
# 스칼라, 벡터, 메트릭스(행렬), 텐서
#.shape 
# input dim의 개수 = 열의 개수, column, ficture, 특성
print(y.shape) #(10, ) (10행, ) = 10개의 스칼라를 가진 벡터 한개


x=x.T
#T=행열 치환
print(x.shape) #(10,2) 10행 2열

#2. 모델구성

model = Sequential()
model.add(Dense(5,input_dim=2))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))



#3. 컴파일, 훈련

model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=1)

#4. 평가, 예측

loss=model.evaluate(x,y)
print('loss:', loss)

result = model.predict([[10,1.4]])
print('[10,1.4]의 예측 값 : ', result)



"""
결과 : 
"""

