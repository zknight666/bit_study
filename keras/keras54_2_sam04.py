import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten, Input,concatenate, LSTM
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. data 전처리
samdf01_csv=pd.read_csv('C:/study/_data/sam01/삼성전자 주가.csv',header=0,encoding='cp949',thousands=',').loc[::-1]
amodf01_csv=pd.read_csv('C:/study/_data/sam01/아모레퍼시픽 주가.csv', header=0,encoding='cp949', thousands=',').loc[::-1]
print(samdf01_csv.info())
print(amodf01_csv.info())

samdf01_csv=samdf01_csv.dropna()
amodf01_csv=amodf01_csv.dropna()

sam_x = samdf01_csv[['시가', '고가', '저가', '종가', '거래량']]
amo_x = amodf01_csv.loc[1986:0,['시가', '고가', '저가', '종가', '거래량']]
sam_y = samdf01_csv[['시가']].to_numpy()

amo_x.tail()
sam_x.tail()
print(amo_x.shape) # (1977, 5)
print(sam_x.shape) # (1977, 5)
print(sam_y.shape) # (1977, 1)

sam_x = MinMaxScaler().fit_transform(sam_x)
amo_x = MinMaxScaler().fit_transform(amo_x)

def split_data(dataset, timesteps):
    tmp = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        tmp.append(subset)
    return np.array(tmp)

sam_x = split_data(sam_x,1)
amo_x = split_data(amo_x, 1)
# print(samsung_x.shape) #(1976, 5, 5)
# print(amore_x.shape) #(1976, 5, 5)

sam_y = sam_y[4:, :] # x 데이터와 shape을 맞춰주기 위해 4개 행 제거
# print(samsung_y.shape) #(1976, 1)

sam_x_predict = sam_x[-1].reshape(-1, 5, 5)
amo_x_predict = amo_x[-1].reshape(-1, 5, 5)

x1_train, x1_test, y_train, y_test = train_test_split(sam_x, sam_y, train_size=0.8,shuffle=False)

x2_train, x2_test = train_test_split(amo_x, train_size=0.8,shuffle=False)




print("______________")
print(x1_train.shape) # (1581, 5)
print(x1_test.shape) # (396, 5)
print(x2_train.shape) # (1581, 5)
print(x2_test.shape) # (396, 5)

# 삼성전자
samsung_input = Input(shape=(5, 5))
samsung_layer1 = LSTM(10, return_sequences=True,activation='relu')(samsung_input)
samsung_layer1 = Dropout(0.2)(samsung_layer1)
samsung_layer2 = LSTM(10, activation='relu')(samsung_layer1)
samsung_layer3 = Dense(12, activation='relu')(samsung_layer2)
samsung_output = Dense(1)(samsung_layer3)

# 아모레퍼시픽
amore_input = Input(shape=(5, 5))
amore_layer1 = LSTM(15, return_sequences=True,activation='relu')(amore_input)
amore_layer1 = Dropout(0.2)(amore_layer1)
amore_layer2 = LSTM(25, activation='relu')(amore_layer1)
amore_layer3 = Dense(15, activation='relu')(amore_layer2)
amore_output = Dense(1)(amore_layer3)

# 병합
merge_layer1 = concatenate([samsung_output, amore_output])
merge_layer2 = Dense(10, activation='relu')(merge_layer1)
merge_output = Dense(1, activation='relu')(merge_layer2)

model = Model(inputs=[samsung_input, amore_input], outputs=[merge_output])

# y_train=y_train.values
# y_test=y_test.values
# y_train=y_train.reshape(1581,)
# y_test=y_test.reshape(396,)
# print(y_train.shape) # (1581,)
# print(y_test.shape) # (396,)


# #2-1 모델1
# input1=Input(shape=(5,))
# dense1=Dense(11,activation='relu', name='ds11')(input1)
# dense2=Dense(12,activation='relu', name='ds12')(dense1)
# dense3=Dense(13,activation='relu', name='ds13')(dense2)
# output1=Dense(10,activation='relu', name='ds14')(dense3)


# #2-2 모델2
# input2=Input(shape=(5,))
# dense21=Dense(21,activation='relu', name='ds21')(input2)
# dense22=Dense(22,activation='relu', name='ds22')(dense21)
# output2=Dense(23,activation='relu', name='ds23')(dense22)


# #2-3 모델 병합
# merge1=concatenate([output1,output2],name='mg1')
# merge2=Dense(10, activation='relu',name='mg2')(merge1)
# merge3=Dense(15,name='mg3')(merge2)
# last_output=Dense(1,name='last')(merge3)

# model=Model(inputs=[input1,input2],outputs=last_output)

# model.summary()




#3. compile, training
model.compile(
    loss='mae',
    optimizer='adam',
    metrics=['mse']
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=2,
    restore_best_weights=True
)


import datetime
date=datetime.datetime.now()


date=date.strftime("%m%d_%H%M")


model_checkpoint=ModelCheckpoint(
    filepath='c:/study/_save/MCP/' + 'k54_sam01_' + date + '_{epoch}-{val_loss:.4f}.h5',
    monitor='val_loss',
    verbose=2,
    save_best_only=True
)

model.fit(
    [x1_train,x2_train],y_train,
    epochs=75752576,
    batch_size=1,
    verbose=2,
    validation_split=0.2,
    callbacks=[early_stopping]
)


#4. 평가, 예측

loss=model.evaluate([x1_test,x2_test],y_test)
y_predict = model.predict([x1_test,x2_test])
print('loss : ',loss)
print("삼성전자 시가 :" , y_predict)

