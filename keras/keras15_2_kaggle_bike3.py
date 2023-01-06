#0. Load libraries

import numpy as np
import pandas as pd # 
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, Model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout


#1. data 전 처리 (결측치 확인, column, index 조절 )

#데이터 불러오기

path = './_data/bike/'

train_csv = pd.read_csv(path+'train.csv',index_col=[0,9,10]) # 날씨, 'Casual','Registared' 컬럼 drop
#train 0, 9,10 column 제거
test_csv = pd.read_csv(path+'test.csv',index_col=0)
sampleSubmission_csv = pd.read_csv(path+'sampleSubmission.csv',index_col=0)


print(train_csv)
print(train_csv.shape) # (10886, 9)

print(test_csv.shape) # (6493, 8)

# print(train_csv.info()) # 결측치 없음

# print(test_csv.info())

x=train_csv.drop('count',axis=1) # 'count','Casual','Registared' 컬럼 제거
print(x) # [10886 rows x 8 columns]

y=train_csv['count']
print(y)

print(y.shape)  # (10886,)

x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    train_size=0.7,
    shuffle=True,
    random_state=9
)

x_test,x_val,y_test,y_val = train_test_split(
    x_test,y_test,
    train_size=0.2,
    random_state=9)


print(x_train.shape,x_test.shape) # (7620, 8) (3266, 8)
print(y_train.shape,y_test.shape) # (7620,) (3266,)






#2. model 구성


model=Sequential()
inputs = Input(shape=(8,))
hidden1 = Dense(64, activation='relu')(inputs)
hidden2 = Dense(64, activation='relu')(hidden1)
output = Dense(1)(hidden2)
model = Model(inputs=inputs, outputs=output)




# model.add(Dense(16,input_dim=8))
# # model.add(Dropout(rate=0.2))
# model.add(Dense(512,activation='relu'))
# # model.add(Dense(256,activation='relu'))
# # model.add(Dense(128,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(1))







#3. compile, training

model.compile(
    optimizer='nadam',
    loss='mae',
    metrics=['accuracy']
)

model.fit(
    x_train,y_train,
    batch_size=32,
    epochs=100,
    validation_data=(x_val,y_val)
)





#4. 평가, 예측

loss=model.evaluate(x_test,y_test)
print('loss : ',loss)


y_predict=model.predict(x_test)


def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print('RMSE : ',RMSE(y_test,y_predict))

r2=r2_score(y_test,y_predict)
print('r2 : ',r2)







#5. 제출
y_submit=model.predict(test_csv)
sampleSubmission_csv['count'] = y_submit
print(sampleSubmission_csv)
sampleSubmission_csv.to_csv(path + 'submission_0106_04.csv')

"""
1) 
mse, batch_size=32, epochs=300 32,512,32 relu
loss :  [23135.67578125, 0.012553581967949867]
RMSE :  152.10417454109475
r2 :  0.31126218056215427

loss :  [23789.810546875, 0.012553581967949867]
RMSE :  154.23945388405724
r2 :  0.29178907514966257

loss :  [22934.822265625, 0.012247397564351559]
RMSE :  151.44246163572885
r2 :  0.31724170566485954

"""