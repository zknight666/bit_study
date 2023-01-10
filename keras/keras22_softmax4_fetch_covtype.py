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




#1. data

datasets=fetch_covtype()
x=datasets.data
y=datasets.target

# print(datasets.get_data_home())

# print(x.shape,y.shape) # (581012, 54) (581012,) x= 54개





print(np.unique(y,return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64)) y=7개
# 데이터 한개 2700개이므로 stratify 사용
# 함정 있음. 인코딩 다른거 해야할듯? 
print(datasets.feature_names) 
print(x,y,y.shape)

y=to_categorical(y) # 앞에 0이 없으면 0 생김 -> output 8개됨 -> 만약 쓴다면 앞자리 index 제거해주거나 아니면 다른 encoder 써야될듯
y=np.delete(y,1,axis=1)

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


"""

# ohe = OneHotEncoder()
# ohe.fit(y.reshape(-1,1))
# y=ohe.transform(y.reshape(-1,1)).toarray()

# y=pd.get_dummies(y)
# y = y.values # 이거 뭔데

print(x)
print(y)
print(x.shape) 
print(y.shape) # (581012, 7)






x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    train_size=0.8,
    random_state=1,
    shuffle=True,
    stratify=y
)




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
print('accuracy : ',accuracy)
print('y_test : ',y_test)
print('y_predict : ',y_predict)
# print('hist : ',hist.history['loss'])
model.summary()