import numpy as np
a = np.array(range(1,101))
x_predict = np.array(range(96,106)) # 예상 y= 100, 106

timesteps1 = 5  # x는 4개 , y는 1개
def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1 ):  # 만약 range(3) = (0,1,2)
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset) # 뒤에 추가로 쳐 박는다.
    return np.array(aaa)
bbb = split_x(a, timesteps1)
ccc = split_x(x_predict, 4)
print(bbb.shape) #(96.5)
print(ccc)

x = bbb[:, :-1]  # [행 , 열]
y = bbb[:, -1]
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.75,shuffle=True ,random_state=321)   #3차원 못 받아드려서 2차원일 때 스플린해줘야 함, train_size의 default = 0.75

x_train = x_train.reshape(72,4,1)
x_test = x_test.reshape(24,4,1)

x_predict = ccc.reshape(7,4,1)

#2.model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

model = Sequential()
model.add(LSTM(64, input_shape=(4,1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#3.compile, fit
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=2000,batch_size=1,validation_split=0.1)

#4. evaluate
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_predict)
print(y_predict)
"""
[[100.50194]
 [101.2628 ]
 [101.97278]
 [102.6306 ]
 [103.23637]
 [103.79155]
 [104.29857]]


"""






