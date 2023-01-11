import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping #, ModelCheckpoint
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1. data (data 불러오기, column 확인, 결측치 확인, 데이터 전처리)
"""
pd.read_csv
"""

path = './_data/ddarung/'

# [1459 rows x 10 columns]
train_csv = pd.read_csv(path+'train.csv', index_col=[0])
test_csv = pd.read_csv(path+'test.csv', index_col=0)  # [715 rows x 9 columns]
submission_csv = pd.read_csv(path+'submission.csv', index_col=0)

print(train_csv)
print(test_csv)
print(submission_csv)

print(train_csv.shape)  # (1459, 10)
print(test_csv.shape)  # (715, 9)
print(submission_csv.shape)  # (715, 1)

print(train_csv.columns)  # column 확인

print(train_csv.info())  # 결측치 확인
print(test_csv.info())
print(train_csv.describe())  # 최대,최소값 확인


print(train_csv.isnull().sum())
train_csv = train_csv.dropna()  # 결측치 삭제

x = train_csv.drop('count', axis=1)
y = train_csv['count']
print(x)
print(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=1,
    shuffle=True
)

# print(x_train.shape, x_test.shape)  # (1167, 9) (292, 9)
# print(y_train.shape, y_test.shape)  # (1167,) (292,)

# scaler_standard=StandardScaler()
# x_train=scaler_standard.fit_transform(x_train)
# x_test=scaler_standard.transform(x_test)
scaler_minmax=MinMaxScaler()
x_train=scaler_minmax.fit_transform(x_train)
x_test=scaler_minmax.transform(x_test)
test_csv=scaler_minmax.transform(test_csv) #x data를 scaling했으므로 submission x data 또한 scaling 필요 / 모든 x data scaling 필요
 



# 2. model
# model = Sequential()
# model.add(Dense(32, input_dim=9))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='selu'))
# # model.add(Dense(50, activation='relu'))
# model.add(Dense(1, activation='relu'))


#2.2 model (함수형)
input1=Input(shape=(9,))
dense1=Dense(32,activation='relu')(input1)
dense2=Dense(512,activation='relu')(dense1)
dense3=Dense(512,activation='relu')(dense2)
output1=Dense(1)(dense3)
model=Model(inputs=input1,outputs=output1)







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







#6. 제출
y_submit=model.predict(test_csv)
print(y_submit)
print(y_submit.shape)

submission_csv['count']=y_submit
print('제출용 csv : ',submission_csv)
submission_csv.to_csv(path+'submission_01.11_02.csv')





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
loss: [33.211708068847656, 2368.175537109375]
RMSE: 48.66390251117757
r2: 0.592475322958559
loss: [35.60508346557617, 2733.531494140625]
RMSE: 52.28318467751592
r2: 0.6370946700750323

loss: [29.216039657592773, 1827.946533203125]
RMSE: 42.754489633842525
r2: 0.7339550512057256

minmax scaler 사용했을 때
loss: [29.245567321777344, 2111.11376953125]
RMSE: 45.94685580751569
r2: 0.6717731809589979
loss: [27.152551651000977, 1663.2928466796875]
RMSE: 40.78348583036207
r2: 0.7137742310275241

standard scaler 사용했을 때
loss: [1753.14892578125, 29.435871124267578]
RMSE: 41.87062112160897
r2: 0.7274279781341645


"""
