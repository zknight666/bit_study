#bias 초기값=0, weight 초기값=random, 부동소수점 방식 사용, 부동소수점 빨라, 부동소수점 
#scaling 제일 큰값을 나눠버림 -> x 최대 값 = 1, 최소 값=0에 가까운 숫자. 0~1사이 값으로 변경. (minmax scaler) (X - MIN) / (MAX-MIN)  레버리지 몇배야
#0과1사이 값=> 소수 -> 부동소수점 연산 특화되어 있음(너무 높은 숫자 때문에 계산 불가능한 것도 해결 가능)
#[8,9,10] 인경우 -> min,max scaler 사용하면 [0,?,1] 됨 
# 단점 : 이상치(outlier)에 너무 많은 영향을 받는다

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 일반적으로 함수 = 소문자 / 클래스 = 대문자


#1. data (data 불러오기, 컬럼 확인, 전처리)

#data 불러오기
dataset=load_boston()

x=dataset.data
y=dataset.target

# scaler_minmax=MinMaxScaler()
# scaler_minmax.fit(x)
# x=scaler_minmax.transform(x)
# # print(type(x)) # <class 'numpy.ndarray'>
# print(np.min(x)) # MinMaxScaler 적용됬는지 np.min,max로 확인 최소값 0
# print(np.max(x)) # 최대값 1로 min,max 제대로 적용 확인 완료


# scaler_standard=StandardScaler()
# scaler_standard.fit(x)
# x=scaler_standard.transform(x)
# print(np.min(x)) # -3.9071933049810337
# print(np.max(y)) # 50.0

"""
scaling은 무조건 train만
range(0~10) shuffle
train=1~8 -> 0~1
test=0,9 -> train기준으로 대략 -0.1 ~ 1.25 됨, scaling을 벗어나는 경우가 있다. (대부분 벗어날듯)
validation,predict도 마찬가지
train=0~0.8



train data 0 ~ 0.8만 훈련.
scaling x가지고 scaling하면 0~1. 평가시 0~1 범위 넘어감. y_predict할때는 이외의 값이 튀어나올 수 있음
실질적으로 validation 1.1?
x_test=scaler_minmax.transform(x_test)
scaling은 무조건 train만
"""

# print(x.shape) # (506, 13)



# column 확인
print(x.shape) # (506, 13)
print(y.shape) # (506,)



# print(dataset.describe()) 왜 안돠

# data 전처리
x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    train_size=0.8,
    random_state=333,
    shuffle=True
)


scaler_minmax=MinMaxScaler()# train data 0 ~ 0.8만 훈련 data 기준 scaling x가지고 scaling하면 0~1 y_predict할때는 이외의 값이 튀어나올 수 있음 
# scaler_minmax.fit(x_train)
# x_train=scaler_minmax.transform(x_train)
x_train=scaler_minmax.fit_transform(x_train)
x_test=scaler_minmax.transform(x_test) # test에는 fit 쓰면 안됨
# print(type(x)) # <class 'numpy.ndarray'>
print(np.min(x)) # MinMaxScaler 적용됬는지 np.min,max로 확인 최소값 0
print(np.max(x)) # 최대값 1로 min,max 제대로 적용 확인 완료




#2. model
model=Sequential()
# model.add(Dense(1,input_dim=13)) # (?,13)
model.add(Dense(50,input_shape=(13,))) # (13,?)
model.add(Dense(500,activation='selu'))
# model.add(Dense(5,activation='selu'))
# model.add(Dense(5,activation='selu'))
# model.add(Dense(5,activation='selu'))
# model.add(Dense(500,activation='relu'))
# model.add(Dense(5,activation='relu'))
# model.add(Dense(5,activation='relu'))
# model.add(Dense(5,activation='relu'))
# model.add(Dense(500,activation='relu'))
# model.add(Dense(5,activation='relu'))
# model.add(Dense(5,activation='relu'))
# model.add(Dense(5,activation='relu'))
# model.add(Dense(1,activation='relu'))
# model.add(Dense(1,activation='relu'))
model.add(Dense(1))





#3. compile, training
model.compile(
    optimizer='nadam',
    loss='mae',
    metrics=['mse']
)


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=2,
    restore_best_weights=True
)

# model_checkpoint=ModelCheckpoint(
#     filepath='./{epoch}-{val_loss:.2f}-{val_accuracy:.2f}.h5',
#     monitor='val_loss',
#     verbose=2,
#     save_best_only=True
# )


hist=model.fit(
    x_train,y_train,
    epochs=1000,
    batch_size=1,
    verbose=2, # 일반적으로 1=모두 보여줌, 0 = 화면 안나옴,2= 생략해서 보여줌, 나머지=epoch 횟수만 보여줌 verbose 0으로 두면 계산속도가 더 많이 빨라짐.
    # verbose 1= 13초 / verbose=0 = 10초
    callbacks=[early_stopping],
    validation_split=0.2
)



#4. 평가, 예측




loss=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)

def RMSE(y_test,y_predict):
    return(np.sqrt(mean_squared_error(y_test,y_predict)))





#5. 파일 만들기 ()









# 결과 값

print('loss: ',loss)
print('RMSE : ',RMSE(y_test,y_predict))
print('r2: ',r2_score(y_test,y_predict))


print('===============================')
print(hist) # <keras.callbacks.History object at 0x0000020D1D7BE520>
print('===============================')
print(hist.history['loss']) # loss, val_loss 변화값 리스트 형태로 저장되어 있음
# {'키(loss)':[value1,value2,3..])}
# 데이터 형태 = 리스트 => ['ㅁ','ㅠ','ㅊ'], 딕셔너리 => {'ㅁ':0,'ㅠ':2,'ㅊ':5} (딕셔너리는 {'키':value} 형태)


plt.figure(
    figsize=(9,6)
    )




plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue',marker='.', label='val_loss') # epoch 순으로 가서 x값 생략해도 됨
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('boston loss')
plt.legend(loc='upper right') # 'upper left'
plt.show()



"""
결과
scaler 없었을 때
loss :  [2.902878761291504, 0.0]
RMSE :  4.0858269073324704
r2 :  0.8310794041312348

scaler 썼을 때 # 좋지 않은 경우도 존재
loss:  [2.4661803245544434, 14.00954818725586]
RMSE :  3.742933349597218
r2:  0.8571605108056923

standard scaler 썼을 때 # 
loss:  [2.4749691486358643, 19.448814392089844]
RMSE :  4.410080274362113
r2:  0.8017025677403397
"""

