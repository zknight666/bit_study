import numpy as np
import pandas as pd # 
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout

#1. data

path = './_data/ddarung/'
train_csv = pd.read_csv(path+'train.csv',index_col=0)
# ★ index_col=0 = 0번째 컬럼 data 무시, id는 column은 data가 아니라서 & 제거 안하면 자동으로 생성됨
# = train_csv = pd.read_csv('./_data/ddarung/train.csv',index_col=0)
test_csv = pd.read_csv(path+'test.csv',index_col=0)
# test_csv = pd.read_csv('./_data/ddarung/test.csv',index_col=0)
# = ★ index_col=0 = 0번째 컬럼 data 무시, id index는 data가 아니라서
submission = pd.read_csv(path+'submission.csv',index_col=0)

print(train_csv)
"""
      hour  hour_bef_temperature  hour_bef_precipitation  hour_bef_windspeed  hour_bef_humidity  hour_bef_visibility  hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5  count
id
3       20                  16.3                     1.0                 1.5               89.0                576.0           0.027           76.0            33.0   49.0
6       13                  20.1                     0.0                 1.4               48.0                916.0           0.042           73.0            40.0  159.0
7        6                  13.9                     0.0                 0.7               79.0               1382.0           0.033           32.0            19.0   26.0
8       23                   8.1                     0.0                 2.7               54.0                946.0           0.040           75.0            64.0   57.0
9       18                  29.5                     0.0                 4.8                7.0               2000.0           0.057           27.0            11.0  431.0
...    ...                   ...                     ...                 ...                ...                  ...             ...            ...             ...    ...
2174     4                  16.8                     0.0                 1.6               53.0               2000.0           0.031           37.0            27.0   21.0
2175     3                  10.8                     0.0                 3.8               45.0               2000.0           0.039           34.0            19.0   20.0
2176     5                  18.3                     0.0                 1.9               54.0               2000.0           0.009           30.0            21.0   22.0
2178    21                  20.7                     0.0                 3.7               37.0               1395.0           0.082           71.0            36.0  216.0
2179    17                  21.1                     0.0                 3.1               47.0               1973.0           0.046           38.0            17.0  170.0

[1459 rows x 10 columns]
"""
print(train_csv.shape)
"""
(1459, 10)
"""
# ★ count data 분리해줘야하므로 실제 data (1459,9)****

print(train_csv.columns)
# ㅁㅁㅁ.columns 중요
"""
Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
        dtype='object')
"""
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

# 결측치 처리 1. 제거#


print(train_csv.isnull().sum()) # 각 컬럼 당 결측치 확인
train_csv=train_csv.dropna() # 결측치 삭제
print(train_csv.shape) # (1328,10), 결측치 128개 삭제 완료


#x_train, y_train 분리
#y 값 분리



x=train_csv.drop('count',axis=1) # count 컬럼 제거
print(x) # [1459 rows x 9 columns]

y=train_csv['count']
print(y)

print(y.shape)  # (1459,)







x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    train_size=0.8,
    shuffle=True,
    random_state=9
)

print(x_train.shape,x_test.shape) #(929,9) (399,9)
print(y_train.shape,y_test.shape) # (929,), (399,)




#2. model
model=Sequential()
model.add(Dense(64,input_dim=9,activation='relu'))
# model.add(Dense(64))
# model.add(Dropout(rate=0.1))
model.add(Dense(600,activation='relu'))
# model.add(Dropout(rate=0.1))
# model.add(Dense(85))
# model.add(Dense(90))
# model.add(Dense(95))
# model.add(Dropout(rate=0.5))
# model.add(Dense(5,activation='relu'))
model.add(Dense(1))


#3. compile, training
import time


model.compile(
    optimizer='nadam',
    loss='mse',
    metrics='accuracy'
)

start=time.time()

model.fit(
    x_train,y_train,
    batch_size=32,
    epochs=500,
    validation_split=0.2
    )
end=time.time()


#4. 평가, 예측

loss=model.evaluate(x_test,y_test)
print('loss : ',loss)


y_predict=model.predict(x_test)

# print(y_predict)

#결측치 해결

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print('RMSE : ',RMSE(y_test,y_predict))

r2=r2_score(y_test,y_predict)
print('r2 : ',r2)

print("걸린시간",end-start)


"""
결과
epochs 2000
loss :  [2973.72265625, 0.0]
RMSE :  54.53185213064767
r2 :  0.5620099274614669

epochs 2000, dense 1000개
loss :  [3020.390869140625, 0.0]
RMSE :  54.958083763430736
r2 :  0.5551363370526385

cpu 걸린 시간 : 걸린시간 98.47732639312744
GPU 걸린 시간 : ㅁㅁ초


"""

#제출용

# .to_csv()를 사용해서
#submission_0105.csv를 완성하시오


# y_submit=model.predict(test_csv)

# print(y_submit)
# print(y_submit.shape) #(715,1)

# submission['count'] = y_submit
# print(submission)
# submission.to_csv(path + 'submission_01051_13.csv')

#10 = 46
#11 = 43.3
#12 = 42.7
#13 = 

#index 처리, 아직 날짜는 data로 보지 않을 것, 
#