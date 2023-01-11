# 일요일 케글 찍어서 보낼 것 케글 submit 결과, 그 밑에 깃허브 주소 적을 것

# 0. Load libraries

import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt





# 1. data 처리 (결측치 확인, column, index 조절 )

# 데이터 불러오기
path = './_data/bike/'

# 날씨, 'Casual','Registared' 컬럼 제외
train_csv = pd.read_csv(path+'train.csv', index_col=[0, 9, 10])
test_csv = pd.read_csv(path+'test.csv', index_col=0)
sampleSubmission_csv = pd.read_csv(path+'sampleSubmission.csv', index_col=0)


# 데이터 확인, column 확인
print(train_csv)  # datetime, casual, registereddrop 제외 확인 완료
print(test_csv)
print(train_csv.shape)  # (10886, 9)
print(test_csv.shape)  # (6493, 8)


# 결측치 확인
print(train_csv.info())  # 결측치 없음
print(test_csv.info())


# column 맞추기
x = train_csv.drop('count', axis=1)  # count 컬럼 제거
print(x)  # count 컬럼 제거 확인 완료 [10886 rows x 8 columns]

y = train_csv['count']
print(y)

print(y.shape)  # (10886,)


# train, test, val data 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=7
)


# data 재 확인
print(x_train.shape, x_test.shape)  # (8708, 8) (2178, 8)
print(y_train.shape, y_test.shape)  # (8708,) (2178,)


# scaler_standard=StandardScaler()
# x_train=scaler_standard.fit_transform(x_train)
# x_test=scaler_standard.transform(x_test)
scaler_minmax=MinMaxScaler()
x_train=scaler_minmax.fit_transform(x_train)
x_test=scaler_minmax.transform(x_test)





# 2. model 구성
# model = Sequential()
# model.add(Dense(500, input_dim=8, activation='relu'))
# # model.add(Dropout(rate=0.9))
# model.add(Dense(500, activation='selu'))
# model.add(Dense(10, activation='relu'))
# # model.add(Dense(50, activation='relu'))
# # model.add(Dense(5, activation='relu'))
# # model.add(Dense(64, activation='relu'))
# # model.add(Dropout(rate=0.5))
# model.add(Dense(1, activation='relu'))

#2.2 model (함수형)

input1=Input(shape=(8,))
dense1=Dense(1,activation='relu')(input1)
dense2=Dense(1,activation='relu')(dense1)
dense3=Dense(1,activation='relu')(dense2)
dense4=Dense(1,activation='relu')(dense3)
output1=Dense(1)(dense4)
model=Model(inputs=input1,outputs=output1)







# 3. compile, training
model.compile(
    optimizer='adam',
    loss=['mse'],
    metrics=['mae']
)

early_stopping = EarlyStopping(
    monitor='val_loss', 
    verbose=2, 
    patience=100, 
    restore_best_weights=True
    )

# mc = ModelCheckpoint('best_model.h5', monitor='val_loss',
#                      mode='min', save_best_only=True)


hist=model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=75752576,
    validation_split=0.2,
    verbose=2,
    callbacks=[early_stopping]
)


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


r2 = r2_score(y_test, y_predict)

print('r2 : ', r2)
print('RMSE : ', RMSE(y_test, y_predict))
print('loss : ', loss)
print('hist : ', hist.history['loss'])


#5. 시각화 (figsize, plot(data, color, marker, label name), title, x,ylabel,grid,show)

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

#그래프 제목, x축, y축 설정
plt.title('kaggle')
plt.xlabel('epochs')
plt.ylabel('loss')

plt.grid()
plt.legend(loc='upper right')

plt.show()




# 5. 제출 (r2 값 0.317 이상)
# y_submit = model.predict(test_csv)
# sampleSubmission_csv['count'] = y_submit
# # print(sampleSubmission_csv)  # 최종 data 확인
# sampleSubmission_csv.to_csv(path + 'submission_01.11_01.csv')

# r2 값 0.329 이상


"""
결과 값
r2 :  0.32961241803587704
RMSE :  150.3150211496139
loss :  [22594.603515625, 22594.603515625]

r2 :  0.33720117621476986
RMSE :  145.7741478672739
loss :  [21250.103515625, 21250.103515625]


minmax scaler 사용했을 때

standard scaler 사용했을 때
r2 :  0.33875601065886407
RMSE :  145.60306447115434
loss :  [98.92642211914062, 21200.25]
r2 :  0.3337582222243298
RMSE :  146.1522743614668
loss :  [98.355712890625, 21360.486328125]
r2 :  0.3705673077394519
RMSE :  142.0575416918559
loss :  [20180.34375, 104.36688232421875]
"""
