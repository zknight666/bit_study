import numpy as np
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN
from tensorflow.keras.models import Sequential


#1. data

num_data= np.array(range(1,101)) # 1~100
x_predict = np.array(range(96,106)) # 96~105

timesteps = 5 # x= 4개, y=1개
# 실습 : x_predict도 4개씩 자르기, 예상 y=100~106까지 나오도록. split해줘야함



def split_x(dataset, timesteps):  # : <- 함수 시작 의미
    x_split=[] # 빈리스트 생성
    y_split=[]
    for i in range(len(dataset)-timesteps+1): # len(dataset) = dataset의 총 길이 = 100개  #  100-5+1=94번 반복 의미 -> i에 0~93이 들어감    # subset, x_split 문장을 반복
        x_set= dataset[i : (i + timesteps)] # for문을 통해 x_set, x_split 문장을 반복 0:0+5 = 0,1,2,3,4 => 1,2,3,4,5 => 93번 반복 dataset[a:b] => a부터 b까지  / [0:3] -> 1,2,3/ [1:4] -> 2,3,4 / [2:5] -> 3,4,5 까지 3번 반복
        y_set= dataset[i + timesteps] # 0+5 = 5번째 항목 선택 => 6번째,7번째... => 93번 반복
        x_split.append(x_set)
        y_split.append(y_set)
    return np.array(x_split), np.array(y_split)


x_split=split_x(num_data,timesteps) # 
y_split=split_x(x_predict,timesteps)


print(x_split)
print(x_split.shape) # (96, 4)
print(y_split)
print(y_split.shape) # (96,)

x= x_split[:,:-1]
y= y_split[:,-1]

# x=x_split
# y=y_split


# print(x)
# print(y)
# print(x.shape) #(6, 4)
# print(y.shape) # (6, )




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
model.fit(x,y,batch_size=32,epochs=50)


#4. 평가, 예측
results=model.evaluate(x,y)
print('mse : ',results[0])
print('mae : ',results[1])


x_predict=np.array([100,108]).reshape(1,4,1) #
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



