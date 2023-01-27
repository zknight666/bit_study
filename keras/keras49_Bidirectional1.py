#47_2 복붙, 양방향 모델


import numpy as np
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN, Bidirectional
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split


#1. data

num_data= np.array(range(1,101)) # = [1,2,3,4,5]
x_predict= np.array(range(96,106)) # 96~105 예상 y=100,107

timesteps1=5 # x=4, y=1
timesteps2=4 # 

def split_x(dataset, timesteps):  # : <- 함수 시작 의미
    list1=[]    # 빈리스트 생성
    for i in range(len(dataset) - timesteps +1): # len(dataset) = dataset의 리스트  #  5-3+1=3번 반복 의미 -> i에 0,1,2이 들어감    # subset, list1 문장을 반복
        subset= dataset[i : (i + timesteps)] # for문을 통해 subset, list1 문장을 반복       dataset[a:b] => a부터 b까지  / [0:3] -> 1,2,3/ [1:4] -> 2,3,4 / [2:5] -> 3,4,5 까지 3번 반복
        list1.append(subset)
    return np.array(list1)
   
bbb=split_x(num_data,timesteps1)
print(bbb)
print(bbb.shape) # (96,5)


x= bbb[:,:-1] # 마지막 자리를 제외하고 출력
y= bbb[:,-1] # 마지막 자리만 출력

print(x)
print(y)
print(x.shape) #(96, 4)
print(y.shape) # (96, )



x_predict1=split_x(x_predict,timesteps2)
print(x_predict1)
print(x_predict1.shape) # (96,5)



x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    random_state=1,
    shuffle=True,
    train_size=0.75
)

print(x_train.shape) # (72, 4)
print(x_test.shape) # (24, 4)
print(y_test.shape)
print(y_train.shape)
print(x_predict1.shape)

x_train=x_train.reshape(72,4,1) #
x_test=x_test.reshape(24,4,1) #
x_predict=x_predict1.reshape(7,4,1)

print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train.shape)
print(x_predict1.shape)



#2. model
model=Sequential()
# model.add(LSTM(units=100,input_shape=(4,1),activation='relu',return_sequences=True))
model.add(Bidirectional(LSTM(units=100,activation='relu',return_sequences=True),input_shape=(4,1)))
model.add(Bidirectional(LSTM(10)))
#input = (n,3,1) -> 출력 -> (n,10) 로 출력 -> LSTM은 3차원 받아야함 -> 3차원 변환 필요 -> return_sequences=True -> LSTM 반복 가능
# model.add(LSTM(10))
model.add(Dense(300,activation='relu'))
model.add(Dense(1))
model.summary()



#3. compile, training
model.compile(optimizer='adam',loss='mse',metrics='mae')
model.fit(x_train,y_train,batch_size=1,epochs=50,validation_split=0.2)



#4. 평가, 예측
results=model.evaluate(x_test,y_test)
print('mse : ',results[0])
print('mae : ',results[1])


# x_predict=np.array([8,9,10,11]).reshape(1,4,1) #
result=model.predict(x_predict)
print('y=100,107 : ',result)

#  [ 99.9545 ]
#  [100.42926]
#  [100.80662]
#  [101.10367]
#  [101.33633]
#  [101.51834]]

#  [97.42797 ]
#  [97.88982 ]
#  [98.29246 ]
#  [98.64197 ]
#  [98.944336]
#  [99.20518 ]]
#???
