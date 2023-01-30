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

x3_datasets = np.array([range(100,200),range(1301,1401)]).T

y1 = np.array(range(2001,2101)) # 삼성전자의 하루 뒤 종가라 생각
print(y1.shape) # (100,)

y2 = np.array(range(201,301)) # 아모레 하루 뒤 종가

y3 = np.array(range(101,201))



from sklearn.model_selection import train_test_split




# 3개 넣을 수 있다. 각각 어떻게 분리되는지는 print로 확인할 것 # 줄바꾸기 = \
x1_train,x1_test,x2_train,x2_test,x3_train,x3_test,\
y1_train,y1_test,y2_train,y2_test=train_test_split(
    x1_datasets,x2_datasets,x3_datasets,y1,y2,
    train_size=0.7,
    random_state=1234,
)




print("______________")
print(x3_train.shape)
print(x3_train.shape)
print(x1_train.shape)
print(x2_train.shape)
print(y1_train.shape)
print(x1_train.shape)
print(x2_test.shape)
print(y1_test.shape)
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
output1=Dense(14,activation='relu', name='output1')(dense3)


#2-2 모델2
input2=Input(shape=(3,))
dense21=Dense(21,activation='relu', name='ds21')(input2)
dense22=Dense(22,activation='relu', name='ds22')(dense21)
output2=Dense(23,activation='relu', name='output2')(dense22)


#2-3 모델3
input3=Input(shape=(2,))
dense31=Dense(21,activation='relu', name='ds31')(input3)
dense32=Dense(22,activation='relu', name='ds32')(dense31)
output3=Dense(23,activation='relu', name='output3')(dense32)


#2-4 모델 병합
from tensorflow.keras.layers import concatenate, Concatenate
merge1=Concatenate()([output1,output2,output3])
merge2=Dense(12, activation='relu',name='mg2')(merge1)
merge3=Dense(13,name='mg3')(merge2)
last_output=Dense(1,name='last')(merge3)
# model1, model2 = Model(inputs=merged_model.input, outputs=merged_model.layers[-2].output), Model(inputs=merged_model.input, outputs=merged_model.layers[-1].output)

#2-5 모델5 분기 1
dense51=Dense(21,activation='relu')(last_output)
dense52=Dense(22,activation='relu')(dense51)
output5=Dense(23,activation='relu')(dense52)

#2-5 모델5 분기 2
dense61=Dense(21,activation='relu')(last_output)
dense62=Dense(22,activation='relu')(dense61)
output6=Dense(23,activation='relu')(dense62)

model=Model(inputs=[input1,input2,input3],outputs=[output5,output6])

model.summary()







#3. compile, training
model.compile(
    loss='mse',
    optimizer='adam',
)

model.fit(
    [x1_train,x2_train,x3_train],[y1_train,y2_train],
    epochs=50,
    batch_size=1,
    verbose=2,
    validation_split=0.2
)

#4. 평가, 예측
loss=model.evaluate([x1_test,x2_test,x3_test],[y1_test,y2_test])



print('loss : ',loss)


