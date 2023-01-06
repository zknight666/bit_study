import numpy as np
import pandas as pd # 
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#1. data

path = './_data/ddarung/'
train_csv = pd.read_csv(path+'train.csv',index_col=0)
# train_csv = pd.read_csv('./_data/ddarung/train.csv',index_col=0)
test_csv = pd.read_csv(path+'test.csv',index_col=0)
# test_csv = pd.read_csv('./_data/ddarung/test.csv',index_col=0)
# index_col=0 = 0번째 컬럼 data 무시, index는 data가 아니라서
submission = pd.read_csv(path+'submission.csv',index_col=0)

print(train_csv)
print(train_csv.shape) #(1459,10)
# count data 분리해줘야하므로 실제 data (1459,9)

print(train_csv.columns)
# ㅁㅁㅁ.columns 중요
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

print(train_csv.info()) # 결측치, 측정 안된 것 총 data 1459 이지만 최소 1342밖에 없는 것도 있음

"""
Data columns (total 10 columns): 
#   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   hour                    1459 non-null   int64
 1   hour_bef_temperature    1457 non-null   float64
 2   hour_bef_precipitation  1457 non-null   float64
 3   hour_bef_windspeed      1450 non-null   float64
 4   hour_bef_humidity       1457 non-null   float64
 5   hour_bef_visibility     1457 non-null   float64
 6   hour_bef_ozone          1383 non-null   float64
 7   hour_bef_pm10           1369 non-null   float64
 8   hour_bef_pm2.5          1342 non-null   float64
 9   count                   1459 non-null   float64 
 """
# 결측치 있는 data 삭제

print(test_csv.info())
"""
Data columns (total 9 columns):
 #   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   hour                    715 non-null    int64
 1   hour_bef_temperature    714 non-null    float64
 2   hour_bef_precipitation  714 non-null    float64
 3   hour_bef_windspeed      714 non-null    float64
 4   hour_bef_humidity       714 non-null    float64
 5   hour_bef_visibility     714 non-null    float64
 6   hour_bef_ozone          680 non-null    float64
 7   hour_bef_pm10           678 non-null    float64
 8   hour_bef_pm2.5          679 non-null    float64

"""


print(train_csv.describe())
"""
hour  hour_bef_temperature  hour_bef_precipitation  hour_bef_windspeed  hour_bef_humidity  hour_bef_visibility  hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5        count
count  1459.000000           1457.000000             1457.000000         1450.000000        1457.000000          1457.000000     1383.000000    1369.000000     1342.000000  1459.000000
mean     11.493489             16.717433                0.031572            2.479034          52.231297          1405.216884        0.039149      57.168736       30.327124   108.563400
std       6.922790              5.239150                0.174917            1.378265          20.370387           583.131708        0.019509      31.771019       14.713252    82.631733
min       0.000000              3.100000                0.000000            0.000000           7.000000            78.000000        0.003000       9.000000        8.000000     1.000000
25%       5.500000             12.800000                0.000000            1.400000          36.000000           879.000000        0.025500      36.000000       20.000000    37.000000
50%      11.000000             16.600000                0.000000            2.300000          51.000000          1577.000000        0.039000      51.000000       26.000000    96.000000
75%      17.500000             20.100000                0.000000            3.400000          69.000000          1994.000000        0.052000      69.000000       37.000000   150.000000
max      23.000000             30.000000                1.000000            8.000000          99.000000          2000.000000        0.125000     269.000000       90.000000   431.000000
"""
#x_train, y_train 분리
#y 값 분리



x=train_csv.drop('count',axis=1) # count 컬럼 제거
print(x) # [1459 rows x 9 columns]

y=train_csv['count']
print(y)

print(y.shape)  # (1459,)

x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    train_size=0.7,
    shuffle=True,
    random_state=1
)

print(x_train.shape,x_test.shape) #(1021,9) (438,9)
print(y_train.shape,y_test.shape) # (1021,), (438,)




#2. model

# model=Sequential()
# model.add(Dense(100,input_dim=9, activation='relu'))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(1))

model=Sequential()
model.add(Dense(5,input_dim=9, activation='relu'))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))



#3. compile, training

model.compile(
    optimizer='nadam',
    loss='mse',
    metrics='accuracy'
)

model.fit(
    x_train,y_train,
    batch_size=32,
    epochs=100
)




# model.compile(
#     loss='mae',
#     optimizer='nadam',
#     metrics=['accuracy']
#     )

# model.fit(
#     x_train,y_train,
#     batch_size=16,
#     epochs=300
# )




#4. 평가, 예측

loss=model.evaluate(x_test,y_test)


y_predict=model.predict(x_test)

#결측치 해결

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print('RMSE : ',RMSE(y_test,y_predict))

r2=r2_score(y_test,y_predict)
print('r2 : ',r2)












# loss=model.evaluate(x_test,y_test)
# #뭘까
# print('loss : ',loss)

# y_predict=model.predict(x_test)
# #뭘까

# from sklearn.metrics import mean_squared_error
# def RMSE(y_test,y_predict):
#         return np.sqrt(mean_squared_error(y_test,y_predict))

# print("RMSE : ",RMSE(y_test,y_predict))

# from sklearn.metrics import r2_score
# R2 = r2_score(y_test,y_predict)
# print("R2 : ",R2)

#제출용

y_submit=model.predict(test_csv)

























