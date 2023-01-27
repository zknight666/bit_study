# x 3,5 -> 1개
# y한칸 밀림
# 앙상블 = 1개모델 + 1개모델 -> 병합 -> 1개 새로운 모델  (총 3개 모델 사용)
# 삼성 모델 + 아모레 모델 -> 새 모델 -> 삼성 00일 종가는?
#sequential 모델 사용 어려움 / 함수형 모델 사용할 것

import numpy as np
x1_datasets = np.array([range(100),range(301,401)])
# transform 필요
print(x1_datasets.shape) #(2, 100)
#-----------------
x1_datasets = np.array([range(100),range(301,401)]).transpose() # 삼성전자의 시가 ,종가 data라 생각
print(x1_datasets.shape) #(100, 2)

x2_datasets = np.array([range(101,201),range(411,511),range(150,250)]).T # 아모레의 3개 컬럼이라 생각
print(x2_datasets.shape) #(100, 3)


y = np.array(range(2001,2101)) # 삼성전자의 하루 뒤 종가라 생각
print(y.shape) # (100,)


from sklearn.model_selection import train_test_split

# 3개 넣을 수 있다. 각각 어떻게 분리되는지는 print로 확인할 것
x1_train,x1_test,x2_train,x2_test,y_train,y_test=train_test_split(
    x1_datasets,x2_datasets,y,
    train_size=0.7,
    random_state=1234,
)
print("______________")
print(x1_train.shape)
print(x2_train.shape)
print(y_train.shape)
print(x1_train.shape)
print(x2_test.shape)
print(y_test.shape)
# (70, 2)
# (70, 3)
# (70,)
# (70, 2)
# (30, 3)
# (30,)








# model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1 모델1
input1=Input(shape=(2,))
dense1=Dense(11,activation='relu', name='ds11')(input1)
dense2=Dense(12,activation='relu', name='ds12')(dense1)
dense3=Dense(13,activation='relu', name='ds13')(dense2)
output1=Dense(14,activation='relu', name='ds14')(dense3)


#2-2 모델2
input2=Input(shape=(3,))
dense21=Dense(21,activation='linear', name='ds21')(input2)
dense22=Dense(22,activation='linear', name='ds22')(dense21)
output2=Dense(23,activation='linear', name='ds23')(dense22)
# dense24=Dense(24,activation='relu', name='ds14')(dense23)


#2-3 모델 병합
from tensorflow.keras.layers import concatenate
merge1=concatenate([output1,output2],name='mg1')
merge2=Dense(12, activation='relu',name='mg2')(merge1)
merge3=Dense(13,name='mg3')(merge2)
last_output=Dense(1,name='last')(merge3)

model=Model(inputs=[input1,input2],outputs=last_output)

model.summary()







#3. compile, training
model.compile(
    loss='mse',
    optimizer='nadam',
    metrics=['mae']
)

model.fit(
    [x1_train,x2_train],y_train,
    epochs=50,
    batch_size=1,
    verbose=2,
    validation_split=0.2
)

#4. 평가, 예측
loss=model.evaluate([x1_test,x2_test],y_test)

print('loss : ',loss)










