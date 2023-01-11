from sklearn.datasets import load_iris # 꽃잎 넓이 길이. 줄기 넓이 길이 -> 어떤 꽃인지 예측하는 datasets
import numpy as np
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#1. data





datasets=load_iris()

# print(datasets.DESCR) #☆ pandas .describe() / .info()
#class 꽃의 종류 x 컬럼4개, y 컬럼 1개(안에 3가지 클래스)
#class correlation = 상관관계, 
# print(datasets.feature_names) # ☆ pandas .columns

x=datasets.data
y=datasets['target']

print(x)
print(y)
print(x.shape) # (150, 4)
print(y.shape) # (150,) 0,1,2 로 되있는거 보니 labelencorder 되있는듯? -> 결국 encoding했넹 / # y클래스 개수만큼 columns
# one hot encoding 잡아줘야함
"""
encoding하는 다양한 방법
1) from tensorflow.keras.utils import to_categorical
2) # dummy = pd.get_dummies(y)
3) from sklearn.preprocessing import OneHotEncoder
"""

y=to_categorical(y)

print(y)
print(y.shape) # (150,) -> (150,3) 으로 바뀜 input_dim=4, output_dim=3

# ohe = OneHotEncoder()
# ohe.fit(y.reshape(-1,1))
# y=ohe.transform(y.reshape(-1,1)).toarray()

# print(y)
# dummy = pd.get_dummies(y.reshape(-1,1))





x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    random_state=1,
    train_size=0.8,
    shuffle=True, # 결국 랜덤이기 때문에 특정 값에 쏠리는 현상이 발생 하는 경우가 생김. 특정 클래스 배제 현상
    stratify=y # 분류에서만 사용 가능 ★ (y 클래스 2개 이상 존재해야함)
)

print(y_train)
print(y_test)

# scaler_minmax=MinMaxScaler()
scaler_standard=StandardScaler()
x_train=scaler_standard.fit_transform(x_train)
x_test=scaler_standard.transform(x_test)
# x_train=scaler_minmax.fit_transform(x_train)
# x_test=scaler_minmax.transform(x_test)



#2. model
model=Sequential()
model.add(Dense(50,activation='relu', input_shape=(4,)))
model.add(Dense(50))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='linear'))
model.add(Dense(3,activation='softmax')) # class가 3개임, 다중분류 => softmax 사용, node 3개 사용 / softmax 원리 각 클래스별 확률 계산 총합 = 1



#3. compile, training
model.compile(
    loss='categorical_crossentropy', # 다중분류 => categorical_crossentropy 사용
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    x_train,y_train,
    epochs=100,
    batch_size=1,
    verbose=2,
    validation_split=0.2
)





#4. 평가, 예측
loss, accuracy=model.evaluate(x_test,y_test) # accuracy argmax 적용한 결과
y_predict=model.predict(x_test)
# y_predict=model.predict(x_test[:5])
# print(y_test[:5])
# print(y_predict) # accuracy argmax 안쓰고 원값 나옴

print('loss : ',loss)
print('accuracy : ',accuracy) # evaluate에서 쓴 acc와, acc_score랑 같음




print('y_predict(예측값)',y_predict) # 원래 모양
print('y_test(원래값)',y_test) # 원래 모양



y_predict=np.argmax(y_predict,axis=1) # argmax 사용한 모양 가장 높은 값 뽑아줌
print('y_predict(예측값)',y_predict)
y_test=np.argmax(y_test,axis=1) # 가장 높은 값 뽑아줌
print('y_test(원래값)',y_test)





acc=accuracy_score(y_test,y_predict) # y predict는 실수값, y_test는 정수값
#argmax사용

print(acc)


"""
안사용

scaler 사용
loss :  2.9007529178670666e-07
accuracy :  1.0

"""








