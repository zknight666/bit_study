import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten

#1. data 전처리
# 1.1) data 불러오기

train_csv=pd.read_csv('C:/study/_data/ddarung/train.csv',index_col=[0])
test_csv=pd.read_csv('C:/study/_data/ddarung/test.csv',index_col=[0])
submission_csv=pd.read_csv('C:/study/_data/ddarung/submission.csv',index_col=[0])


#1.2) 총 data 개수, 컬럼 확인 & 클래스 확인 & Dtype 확인 & 결측치 확인
print(train_csv.shape) # (1459, 10)
print(test_csv.shape) # (715, 9)
print(train_csv.info()) # <class 'pandas.core.frame.DataFrame'>, 10개 컬럼, 결측치 존재, dtypes: float64(9), int64(1)
print(test_csv.info()) # <class 'pandas.core.frame.DataFrame'>, 9개 컬럼, 결측치 존재, dtypes: float64(8), int64(1)

#1.3) 결측치 삭제
print(train_csv.isnull().sum()) # 각 컬럼 별 결측치 개수 따로 확인
train_csv=train_csv.dropna() # 결측치 있는 행 삭제

print(train_csv.isnull().sum()) # 결측치 삭제 여부 재 확인
print(train_csv.info()) #data 개수 재 확인 (결측치를 삭제했을 때 각 컬럼 별 data가 충분한지) => 한 컬럼당 1328개 data
print(train_csv.shape)
# test의 결측치는 안지우나?


#1.4) x 나누기 (submission용 컬럼 drop 시키기) & column 맞추기(# test에 없는 train 컬럼 drop시키기)
x=train_csv.drop('count',axis=1) # [axis=1 => columns / axis=0 => rows] 의미
y=train_csv['count']

#1.5) column drop 여부 확인
print(x.shape) # (1328, 9)
print(y.shape) # (1328,)


#1.6) train, test split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    random_state=1234,
    train_size=0.8
)

#1.7) data 분류 후 train, test data 양 확인
print(x_train.shape) # (1062, 9)
print(x_test.shape) # (266, 9)


#1.8) CNN 모델을 위한 reshape를 위한 numpy 변환
x_train=x_train.values
x_test=x_test.values


# #1.9) scaler 적용 ★ 차원 다르면 적용 안되므로 순서 중요 (numpy 변환 후 reshape 전)★
# scaler_minmax=MinMaxScaler()
# x_train=scaler_minmax.fit_transform(x_train)
# x_test=scaler_minmax.transform(x_test)
# test_csv=scaler_minmax.transform(test_csv)

#1.10) CNN 모델을 위한 reshape
x_train = np.reshape(x_train, (1062, 9, 1,1))
x_test = np.reshape(x_test, (266, 9, 1,1))

#1.10) reshape 확인
print(x_train.shape) # (1062, 9, 1, 1)
print(x_test.shape) # (266, 9, 1, 1)




#2. CNN 모델
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(2,1),strides=1,padding='same',activation='relu',input_shape=(9,1,1),))
model.add(Conv2D(filters=32,kernel_size=(2,1),strides=1,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,1),strides=1))
model.add(Dropout(rate=0.2))
model.add(Dense(128,activation='relu'))
model.add(Flatten())
model.add(Dense(1,activation='relu'))

model.summary() # Total params: 7,425



#3. compile, training









# 4. 평가, 예측







"""


"""








