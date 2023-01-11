from sklearn.datasets import fetch_covtype, get_data_home
import numpy as np
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler




#1. data

datasets=fetch_covtype()
x=datasets.data
y=datasets.target

# print(datasets.get_data_home())

print(np.unique(y,return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64)) y=7개
# 데이터 한개 2700개이므로 stratify 사용
# 함정 있음. 인코딩 다른거 해야할듯? 
print(datasets.feature_names)
print(x,y,y.shape)
"""
encoding을 하는 3가지 방법
1)
y=pd.get_dummies(y)
y = y.values # 이거 뭔데

2) 
ohe = OneHotEncoder()
ohe.fit(y.reshape(-1,1))
y=ohe.transform(y.reshape(-1,1)).toarray()

3) 
y=to_categorical(y) ->  # 앞에 0이 없으면 0 생김 -> output 8개됨 -> 만약 쓴다면 앞자리 index 제거해줘야함
y=np.delete(y,1,axis=1)


변환 : .toarray, flattern, .values
"""

"""
---------------------------------y=to_categorical(y) -------------------------------------

y=to_categorical(y) # 앞에 0이 없으면 0 생김 -> output 8개됨 -> 만약 쓴다면 앞자리 index 제거해주거나 아니면 다른 encoder 써야될듯


print(type(y))
print(np.unique(y[:,0], return_counts=True)) # 해당 코드 의미 : 0번째 인덱스를 보여라 # (array([0.], dtype=float32), array([581012], dtype=int64)) => 0만 존재하는것 확인함
print(np.unique(y[:,1], return_counts=True)) # (array([0., 1.], dtype=float32), array([369172, 211840], dtype=int64)) -> 0,1로 존재해야하는 것 확인함
# int=정수 / float = 실수

y=np.delete(y,0,axis=1) # 1인가 0인가 -> 0 이다
print(np.unique(y[:,0], return_counts=True)) # 0번째 인덱스 제대로 지워진 것 확인함
----------------------------------------------------------------------------------------------
"""

"""
---------------------------------y=pd.get_dummies(y)-------------------------------------
#  numpy 자료형(아래쪽의 argmax가)이 바로 pandas를 못받아들임 -> pandas를 numpy로 변경시 해결됨
y=pd.get_dummies(y)

y = y.values
또는 
y=y.to_numpy() 사용시 해결 됨
print(y.shape)
print(type(y))

----------------------------------------------------------------------------------------------
"""

"""
---------------------------------ohe = OneHotEncoder()-------------------------------------
ohe = OneHotEncoder()
y=y.reshape(581012,1)
ohe.fit(y)
y=ohe.transform(y)
y=y.toarray()
print(y.shape)
print(type(y))

또는
ohe = OneHotEncoder()
y=y.reshape(581012,1)
y=ohe.fit_transform(y)
y=y.toarray()

----------------------------------------------------------------------------------------------

"""


# ohe = OneHotEncoder() #onehotencorder 사용시 무조건 toarray 사용
# print(y.shape) # (581012,) 벡터 1개 = 1차원 -> 2차원 변경
# y=y.reshape(581012,1)
# print(y.shape) # (581012, 1)
# y=ohe.fit_transform(y) # 오류 나옴 1차원으로 나옴 -> 2차원으로 제공해야함 # fit=실행시키다
# # ex) (3,) -> [1,2,3] ->reshape -> [[1],[2],[3]]
# print(y[:15])
# print(y.shape)
# # 의미 : 0번째 행에 4번째가 1, 1번째 행에 4번째가 1, 2번째 행에 1번째가 1.....
# print(type(y)) # <class 'scipy.sparse._csr.csr_matrix'>
# y=y.toarray()
# print(y[:15])
# print(y.shape)
# print(type(y)) # <class 'numpy.ndarray'> numpy 변환 확인 완료



# y=ohe.transform(y.reshape(-1,1)).toarray()

y=pd.get_dummies(y)
print(type(y)) # <class 'pandas.core.frame.DataFrame'> pandas의 dataframe 형태 (index, header형태로 보여줌)
y = y.values  # numpy 자료형이 바로 pandas를 못받아들임 -> pandas를 numpy로 변경시 해결됨
# y=y.to_numpy()






# print(x)
print(y[:10])
print(type(y)) # <class 'numpy.ndarray'>
print(x.shape) 
print(y.shape) # (581012, 7)






x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    train_size=0.8,
    random_state=1,
    shuffle=True,
    stratify=y
)
 
 
# scaler_standard=StandardScaler()
# x_train=scaler_standard.fit_transform(x_train)
# x_test=scaler_standard.transform(x_test)
scaler_minmax=MinMaxScaler()
x_train=scaler_minmax.fit_transform(x_train)
x_test=scaler_minmax.transform(x_test)



#2. model
model=Sequential()
model.add(Dense(32,activation='relu',input_shape=(54,)))
model.add(Dense(400,activation='relu'))
model.add(Dense(7,activation='softmax'))




#3. compile, training
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    
)



early_stopping=EarlyStopping(
    monitor='val_loss',
    patience=25,
    verbose=2,
    restore_best_weights=True
)



hist=model.fit(
    x_train,y_train,
    epochs=500, # 마지막까지 에러 없을때 epochs 정상적으로 사용
    batch_size=250,
    verbose=2,
    validation_split=0.2,
    callbacks=[early_stopping]
)



#4. 평가, 예측
loss, accuracy=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)

y_predict=np.argmax(y_predict,axis=1)

y_test=np.argmax(y_test,axis=1)


print('loss : ',loss)
print('y_test : ',y_test[:20])
print('y_predict : ',y_predict[:20])
print('accuracy : ',accuracy)
# print('hist : ',hist.history['loss'])
model.summary()

"""
결과

안사용
: accuracy :  0.8860958814620972

standard scaler 사용
loss :  0.17394116520881653
accuracy :  0.9326953887939453

minmax scaler 사용
loss :  0.20554733276367188
accuracy :  0.9198471903800964
"""
