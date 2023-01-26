import numpy as np
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN
from tensorflow.keras.models import Sequential



#1. data

a= np.array(range(1,11)) # = [1,2,3,4,5]

timesteps=5   # => (n,5,1)



def split_x(dataset, timesteps):  # : <- 함수 시작 의미
    aaa=[]    # 빈리스트 생성
    for i in range(len(dataset) - timesteps +1): # len(dataset) = dataset의 리스트  #  5-3+1=3번 반복 의미 -> i에 0,1,2이 들어감    # subset, aaa 문장을 반복
        subset= dataset[i : (i + timesteps)] # for문을 통해 subset, aaa 문장을 반복       dataset[a:b] => a부터 b까지  / [0:3] -> 1,2,3/ [1:4] -> 2,3,4 / [2:5] -> 3,4,5 까지 3번 반복
        aaa.append(subset)
    return np.array(aaa)
    # 리스트에 한개씩 넣기
bbb=split_x(a,timesteps)
print(bbb)
print(bbb.shape) # (6,5)


x= bbb[:,:-1]
y= bbb[:,-1]

print(x)
print(y)
print(x.shape) #(6, 4)
print(y.shape) # (6, )


x=x.reshape(6,4,1) #

print(x.shape) #

# 실습
# LSTM 모델 구성



#2. model
model=Sequential()
model.add(LSTM(units=10,input_shape=(4,1), activation='relu',return_sequences=True))
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


x_predict=np.array([8,9,10,11]).reshape(1,4,1) #
result=model.predict(x_predict)
print('11 결과 : ',result)


"""
[[1 2 3]
[2 3 4]
[3 4 5]]
"""
# def split(dataset,time_steps):
#     xs=list()
#     ys=list()
#     for i in range(0,len(dataset)-time_steps):
#         x=dataset[i:i+time_steps]
#         y=dataset[i+time_steps]
#         # print(f"{x}:{y}")
#         xs.append(x)
#         ys.append(y)
#     return np.array(xs),np.array(ys)
# xs,ys=split(dataset,4)
# 이것도 참고해보기


