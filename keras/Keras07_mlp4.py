import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

#1.데이터

x=np.array([range(10)])  #input 1개
"""
print(range(10))
range 10 = 0부터 10전까지
range(21,31) = 21부터 31전까지 (21~30)
 """

y=np.array([[1,2,3,4,5,6,7,8,9,10], 
           [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
           [9,8,7,6,5,4,3,2,1,0]]) # , output 3개



print(x.shape) #
print(y.shape) #
x=x.T
y=y.T
print(x.shape) #
print(y.shape) #


model=Sequential()
model.add(Dense(5,input_dim=1)) #input 1개
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(3)) # output 3개

model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=200,batch_size=2)


loss=model.evaluate(x,y)
print('loss : ',loss)

result = model.predict([[9]])
print('predict : ',result)


"""
결과 : predict :  [[5.94541   1.5666394]]

epochs 200 : predict :  [[10.195384   1.6747473]]

"""



