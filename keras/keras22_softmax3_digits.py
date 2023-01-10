import numpy as np
from sklearn.datasets import load_digits

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt







#1. data
datasets=load_digits()
x=datasets.data
y=datasets['target']

print(x.shape,y.shape) # (1797, 64) (1797,) -> input 64개

print(np.unique(y,return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64)) -> output 10개

# plt.gray()
# plt.matshow(datasets.images[9])
# plt.show() # 1700(행) * (8 * 8)(열) * RGB 
# one hot encoding 10개

# 이미지 건들기

y=to_categorical(y)
print(x)
print(y)
print(y.shape) # (1797, 10) 변환 확인 완료






x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    random_state=1,
    shuffle=True,
    train_size=0.8,
    stratify=y
)









#2. model
model=Sequential()
model.add(Dense(50,activation='relu',input_shape=(64,)))
model.add(Dense(40,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='softmax'))








#3. compile, training
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    
)



early_stopping=EarlyStopping(
    monitor='val_loss',
    patience=50,
    verbose=2,
    restore_best_weights=True
)



hist=model.fit(
    x_train,y_train,
    epochs=500,
    batch_size=32,
    verbose=2,
    validation_split=0.2,
    callbacks=[early_stopping]
)





#4. 평가 , 예측
loss, accuracy=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)


print('loss : ',loss)
print('accuracy : ',accuracy)

print(y_test)
print(y_predict)



y_predict=np.argmax(y_predict,axis=1)
y_test=np.argmax(y_test,axis=1)

print('y_test : ',y_test)
print('y_predict : ',y_predict)
print('hist : ',hist.history['loss'])




#5. 시각화
plt.figure(
    figsize=(12,12)
    )
plt.plot(
    hist.history['loss'],
    c='red',
    marker='.',
    laber='loss'
)
# plt.plot(
#     hist.history['accuracy'],
#     c='blue',
#     marker='.',
#     laber='accuracy'
# )
plt.xlabel('epochs')
plt.title('digits')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.grid()
plt.show()




