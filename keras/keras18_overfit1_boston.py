from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

# 함수 클래스 차이 대문자, 소문자??


#1. data (data 불러오기, 컬럼 확인, 전처리)

#data 불러오기
dataset=load_boston()

x=dataset.data
y=dataset.target

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


hist=model.fit(
    x_train,y_train,
    epochs=1000,
    batch_size=32,
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
plt.title('보스턴 보스')
plt.legend(loc='upper right') # 'upper left'
plt.show()


#과제 plot title 영어말고 한글로 바꾸기? (그냥 하면 한글 깨짐) 2줄로 됨 matplotlib 한글 깨짐 나눔고딕 
# 해결

# print ('설정파일 위치: ', mpl.matplotlib_fname())


