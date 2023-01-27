# 이메일 제목 고정: 한태희 00,000원
#첨부파일 소스, 가중치
# 삼성전자 월요일 시가




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten, LSTM
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time


#1. data 전처리

# 1.1) data 불러오기

kospi_csv=pd.read_csv('C:/study/_data/kospi200/kospi200.csv',index_col=[0],encoding='cp949')
samsung_csv=pd.read_csv('C:/study/_data/kospi200/samsung.csv',index_col=[0],encoding='cp949')
# submission_csv=pd.read_csv('C:/study/_data/kospi200/submission.csv',index_col=[0])


#1.2) 총 data 개수, 컬럼 확인 & 클래스 확인 & Dtype 확인 & 결측치 확인
print(kospi_csv.shape) # (426, 5)
print(samsung_csv.shape) # (426, 5)
print(kospi_csv.info()) # <class 'pandas.core.frame.DataFrame'>, 5개 컬럼, 결측치 없음, dtypes: float64(4), object(1)
print(samsung_csv.info()) # <class 'pandas.core.frame.DataFrame'>, 5개 컬럼, 결측치 없음, dtypes: object(5)



#1.3) object type -> float64로 변경
kospi_csv = kospi_csv.apply(pd.to_numeric())
samsung_csv = samsung_csv.apply(pd.to_numeric,error='coerce')
print(kospi_csv.info())
print(samsung_csv.info())



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


#1.8) CNN 모델을 위한 reshape를 위한 numpy 변환 / #1.2에서 클래스 pandas 인 것 확인
x_train=x_train.values
x_test=x_test.values

print(type(x_train)) # 변환 확인

# #1.9) scaler 적용 ★ 차원 다르면 적용 안되므로 순서 중요 (numpy 변환 후 reshape 전)★
# scaler_minmax=MinMaxScaler()
# x_train=scaler_minmax.fit_transform(x_train)
# x_test=scaler_minmax.transform(x_test)
# test_csv=scaler_minmax.transform(test_csv)

#1.10) CNN 모델을 위한 reshape
x_train = np.reshape(x_train, (1062, 9, 1))
x_test = np.reshape(x_test, (266, 9, 1))

#1.10) reshape 확인
print(x_train.shape) # (1062, 9, 1)
print(x_test.shape) # (266, 9, 1)




#2. CNN 모델
model=Sequential()
model.add(LSTM(units=15,input_shape=(9,1), activation='relu',return_sequences=True))
model.add(LSTM(15))
model.add(Dense(300,activation='relu')) #
model.add(Dropout(rate=0.3))
model.add(Dense(1,activation='relu'))

model.summary() #



# 3. compile, training

model.compile(
    optimizer='nadam',
    loss='mae',
    metrics=['mse']
)

start = time.time()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    verbose=2,
    restore_best_weights=True
)

import datetime
date=datetime.datetime.now()
date=date.strftime("%m%d_%H%M")

model_checkpoint=ModelCheckpoint(
    filepath='c:/study/_save/MCP/' + 'K31_ddaung_' + date + '_{epoch}-{val_loss:.4f}.h5',
    monitor='val_loss',
    verbose=2,
    save_best_only=True
)




hist=model.fit(
    x_train, y_train,
    epochs=75752576,
    batch_size=4,
    verbose=2,
    validation_split=0.2,
    callbacks=[early_stopping]
)

end = time.time()





# 4. 평가, 예측 (평가 산식 : RMSE)

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)


def RMSE(y_test, y_predict):
    return(np.sqrt(mean_squared_error(y_test, y_predict)))





#7. 결과 확인 (RMSE 48.6 이하로 나올 것)

print('loss:', loss)
print('RMSE:', RMSE(y_test, y_predict))
print('r2:', r2_score(y_test, y_predict))
print('걸린시간 : ', end-start)
# print('hist : ',hist.history['loss'])




#5. 시각화

plt.figure(
    figsize=(9,6)
)

plt.plot(
    hist.history['loss'],
    c='red',
    marker='.',
    label='loss'
)


plt.plot(
    hist.history['val_loss'],
    c='blue',
    marker='.',
    label='val_loss'
)

plt.grid()

plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('ddarung')
plt.legend(loc='upper right')
plt.show()



"""


"""
