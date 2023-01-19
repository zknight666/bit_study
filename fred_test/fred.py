import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import time


# 1. data (data 불러오기, column 확인, 결측치 확인, 데이터 전처리)
datasets = pd.read_csv(
    'C:/Users/hasin/Documents/GitHub/bit_study/_data/fred.stlouisfed/dataset.csv', index_col=[0])
submission_csv = pd.read_csv(
    'C:/Users/hasin/Documents/GitHub/bit_study/_data/fred.stlouisfed/submission.csv', index_col=0)
# datasets = pd.read_csv('C:/study/_data/fred.stlouisfed/dataset.csv', index_col=[0])
# submission_csv = pd.read_csv('C:/study/_data/fred.stlouisfed/submission.csv', index_col=0)


x = datasets.drop('the number of employed people', axis=1)
y = datasets['the number of employed people']

print(x)
print(y)
print(x.shape)  # (266, 6)
print(y.shape)  # (266,)
print(datasets)  # [266 rows x 7 columns]
print(datasets.shape)  # (266, 7)
print(submission_csv)
print(datasets.info())
print(datasets.describe())  # 최대,최소값 확인


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=1,
    shuffle=True
)

# scaler_standard=StandardScaler()
# x_train=scaler_standard.fit_transform(x_train)
# x_test=scaler_standard.transform(x_test)
# scaler_minmax=MinMaxScaler()
# x_train=scaler_minmax.fit_transform(x_train)
# x_test=scaler_minmax.transform(x_test)
# test_csv=scaler_minmax.transform(test_csv)     ##########매우 중요 ### x data를 scaling했으므로 submission x data 또한 scaling 필요 / 모든 x data scaling 필요#######


# 2.2 model (함수형)
input1 = Input(shape=(6,))
dense1 = Dense(32, activation='relu')(input1)
dense2 = Dense(512, activation='relu')(dense1)
dropout1 = Dropout(rate=0.2)(dense2)
dense3 = Dense(512, activation='relu')(dropout1)
dropout2 = Dropout(rate=0.2)(dense3)
output1 = Dense(1)(dropout2)
model = Model(inputs=input1, outputs=output1)


# 3. compile, training

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

start = time.time()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    verbose=2,
    restore_best_weights=True
)

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

model_checkpoint = ModelCheckpoint(
    filepath='c:/study/_save/MCP/' + 'K31_ddaung_' +
    date + '_{epoch}-{val_loss:.4f}.h5',
    monitor='val_loss',
    verbose=2,
    save_best_only=True
)


hist = model.fit(
    x_train, y_train,
    epochs=75752576,
    batch_size=1,
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


# 6. 제출
# y_submit = model.predict(test_csv)
# print(y_submit)
# print(y_submit.shape)

# submission_csv['count']=y_submit
# print('제출용 csv : ',submission_csv)
# submission_csv.to_csv(path+'submission_01.12_01.csv')


# 7. 결과 확인 (RMSE 48.6 이하로 나올 것)

print('loss:', loss)
print('RMSE:', RMSE(y_test, y_predict))
print('r2:', r2_score(y_test, y_predict))
print('걸린시간 : ', end-start)
# print('hist : ',hist.history['loss'])


# 5. 시각화

plt.figure(
    figsize=(9, 6)
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
