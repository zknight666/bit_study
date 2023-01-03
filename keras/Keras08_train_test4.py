import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split 
#1.데이터

x=np.array([range(10), range(21,31), range(201,211)])  #input 3개

y=np.array([[1,2,3,4,5,6,7,8,9,10], 
           [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]]) # (2,10), output 2개


print(x.shape) #(3,10)
print(y.shape) #(2,10)
x=x.T
y=y.T
print(x.shape) #(10,3)
print(y.shape) #(10,2)


#traintestsplit이용하여 7:3으로 잘라서 모델 구현 / 소스 완성

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7, 
    # test_size=0.3, 
    shuffle=True, 
    random_state=1 #사용하면 해당 난수 고정, 사용 안하면 완전 난수 랜덤
    )

print('x_train : ',x_train)
print('x_test : ',x_test)
print('y_train : ',y_train)
print('y_test : ',y_test)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)



model=Sequential()
model.add(Dense(5,input_dim=3)) #input 3개
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(2)) # output 2개

model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=200,batch_size=2)


loss=model.evaluate(x_test,y_test)
print('loss : ',loss)

result = model.predict([[19,40,220]])
print('predict : ',result)


"""
결과 : predict :  [[5.94541   1.5666394]]

epochs 200 : predict :  [[10.195384   1.6747473]]



"""



