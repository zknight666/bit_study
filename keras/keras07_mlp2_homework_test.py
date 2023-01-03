import numpy
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

x=numpy.array([[1,2,3,4],
              [5,6,7,8],
              [9,10,11,12]])
y=numpy.array([20,15,10,5])

print(x.shape) 
# (3,4) -> 3행 4열
    #input_dim = 3 이므로 행열 변환 함수 필요
print(y.shape)

x=x.T
print(x.shape)
# (4,3) -> 4행 3열



model = Sequential()
model.add(Dense(30, input_dim=3))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))


model.compile(loss='mae',optimizer='adam')
model.fit(x,y, epochs=200, batch_size=2)

loss=model.evaluate(x,y)
print('loss : ',loss)

result=model.predict([[4,8,12]])
print('result : ', result)

"""
결과
    loss :  0.23489642143249512
    result :  [[5.191173]]
"""

#test213123123123
#01
