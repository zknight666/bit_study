import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#1. data
datasets=load_breast_cancer()
# x=datasets.data
x=datasets['data'] # load_breast_cancer 안에 들어가 있는 data, target이라는 data
y=datasets['target'] # malignant, benign...

print(x.shape,y.shape) # v(569, 30) (569,)

# print(datasets)
# print(datasets.DESCR) # .DESCR은 sklearn dataset 전용 명령어이므로 안씀, pandas 명령어를 익힐 갓
# print(datasets.feature_names) # feature 30개, 열 30개

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    shuffle=True,
    random_state=7,
    stratify=y
)

# # scaler_minmax=MinMaxScaler()
# scaler_standard=StandardScaler()
# x_train=scaler_standard.fit_transform(x_train)
# x_test=scaler_standard.transform(x_test)
# # x_train=scaler_minmax.fit_transform(x_train)
# # x_test=scaler_minmax.transform(x_test)




#2. model

model=Sequential()
model.add(Dense(50,activation='linear',input_shape=(30,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))










#3. compile, training
model.compile(
    metrics=['acc'],
    loss='binary_crossentropy',
    optimizer='adam'
)


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    verbose=2,
    restore_best_weights=True
)

hist=model.fit(
    x_train,y_train,
    epochs=10000,
    batch_size=16,
    callbacks=[early_stopping],
    validation_split=0.2
)








#4. evaluate, predict

loss,acc=model.evaluate(x_test,y_test)

y_predict=model.predict(x_test)


# sigmoid는 0,1로 출력되는게 아니고 0과 1사이의 값인 실수로 표현되기 때문에 적용 안됨 -> y_predict를 정수형으로 바꾼다.
# 과제 : accuracy를 출력시켜라 (y_predict를 정수형으로 바꿔라)





print('loss : ',loss)
print('acc : ',acc)
# preds_1d = y_predict.flatten() # 차원 펴주기
# pred_class = np.where(preds_1d > 0.5, 1 , 0) #0.5보다크면 2, 작으면 1
# print(classification_report(y_test,pred_class)) # 확인용인듯
# acc2= accuracy_score(y_test,pred_class)
# print('accuracy_score : ',acc2)
#뭐가 다른거야 .. 굳이 accuracy_score를 쓸 필요가?







#5. 시각화
plt.figure(figsize=(9,6))

plt.plot(
    hist.history['val_loss'],
    c='red',
    marker='.',
    label='val_loss'
)

plt.plot(
    hist.history['loss'],
    c='blue',
    marker='.',
    label='loss'
)

plt.title='cancer loss'
plt.xlabel='epochs'
plt.ylabel='val_loss'

plt.grid()
plt.legend(loc='upper right')
plt.show()




#6. 제출

"""
결과 값
scaler 사용안했을 때
loss :  0.1621493101119995
acc :  0.9385964870452881

standard scaler 사용했을 때
loss :  0.059219807386398315
acc :  0.9736841917037964

"""