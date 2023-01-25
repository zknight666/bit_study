import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout
import time
import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt



#1.0) data 불러오기

train_csv=pd.read_csv('C:/study/_data/bike/train.csv',index_col=[0]) # 첫 col 자동 생성되므로 index_col 0 제거 필요.
test_csv=pd.read_csv('C:/study/_data/bike/test.csv',index_col=[0])
submission_csv=pd.read_csv('C:/study/_data/bike/sampleSubmission.csv',index_col=[0])


#1.1) 컬럼 확인 & 클래스 확인 & Dtype 확인 & 결측치 확인
print(train_csv.info()) # <class 'pandas.core.frame.DataFrame'> 11개 컬럼, dtypes: float64(3), int64(8), 결측치 없음
print(test_csv.info()) # 8개 컬럼 (casual, registered, count 값 없음), 결측치 없음

#1.2) x 나누기 (submission용 컬럼 drop 시키기) & column 맞추기(# test에 없는 train 컬럼 drop시키기)
x=train_csv.drop(['count','casual','registered'],axis=1)

#1.3) column drop 여부 확인
print(x) # [10886 rows x 8 columns] 8 columns으로 동일하게 맞춰줌
print(x.shape) # (10886, 8)
#1.4) y 나누기
y=train_csv['count']

#1.5) y 값 확인
print(y)
print(y.shape) # (10886,)


#1.6) train, test split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    shuffle=True,
    random_state=1000,
    train_size=0.8
    
)

#1.7) data 분류 후 train, test data 양 확인
print(x_train.shape) # (8708, 8)
print(x_test.shape) # (2178, 8)

"""
#1.8) CNN 모델을 위한 reshape를 위한 numpy 변환
x_train=x_train.values
x_test=x_test.values

또는

x_train=np.reshape(x_train,(8708,8,1,1))
x_test=np.reshape(x_test,(2178,8,1,1))

np.reshape 사용
"""

#1.8) CNN 모델을 위한 reshape를 위한 numpy 변환
x_train=x_train.values
x_test=x_test.values

# #1.9) scaler 적용 ★ 차원 다르면 적용 안되므로 순서 중요 (numpy 변환 후 reshape 전)★
# scaler_minmax=MinMaxScaler()
# x_train=scaler_minmax.fit_transform(x_train)
# x_test=scaler_minmax.transform(x_test)
# test_csv=scaler_minmax.transform(test_csv)

#1.10) CNN 모델을 위한 reshape
x_train=x_train.reshape(8708,8,1,1)
x_test=x_test.reshape(2178,8,1,1)

#1.10) reshape 확인
print(x_train.shape) # (8708, 8, 1, 1)
print(x_test.shape) # (2178, 8, 1, 1)


#2. CNN 모델
model=Sequential()
model.add(Conv2D(filters=32, kernel_size=(2, 1), input_shape=(8, 1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(rate=0.2))
model.add(Dense(512,activation='relu'))
model.add(Flatten())
model.add(Dense(1,activation='relu'))

#3. compile, training
model.compile(
    optimizer='adam',
    loss='mae',
    metrics=['mse']
)

start = time.time()


early_stopping = EarlyStopping(
    monitor='val_loss', 
    verbose=2, 
    patience=20, 
    restore_best_weights=True
    )

import datetime
date=datetime.datetime.now()
date=date.strftime("%m%d_%H%M")

model_checkpoint=ModelCheckpoint(
    filepath='c:/study/_save/MCP/' + 'K31_kaggle_' + date + '_{epoch}-{val_loss:.4f}.h5',
    monitor='val_loss',
    verbose=2,
    save_best_only=True
)



hist=model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=75752576,
    validation_split=0.2,
    verbose=2,
    callbacks=[early_stopping]
)


end = time.time()



# 4. 평가, 예측
results = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

r2 = r2_score(y_test, y_predict)

print('loss : ', results[0])  # loss와 acc 값 2개 나옴
print('mse : ', results[1])
print('r2 : ', r2)
print('RMSE : ', RMSE(y_test, y_predict))
print('걸린시간 : ', end-start)

"""
loss :  108.44818878173828
mse :  24104.18359375
r2 :  0.21736667377761065
RMSE :  155.25522796854864
걸린시간 :  22.403573751449585

scaler 적용 후
loss :  105.32270050048828
mse :  23031.8984375
r2 :  0.25218252166560307
RMSE :  151.7626391823353
걸린시간 :  96.43893527984619

scaler 적용 전

"""








