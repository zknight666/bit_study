import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
# 1. data


dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape)  # (442, 10)
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=78,
    shuffle=True
)


# 2. model
model = Sequential()
model.add(Dense(5, input_dim=10, activation='selu'))
model.add(Dense(5, activation='selu'))
model.add(Dense(500, activation='selu'))
model.add(Dense(5, activation='selu'))
model.add(Dense(5, activation='selu'))
# model.add(Dense(500,activation='relu'))
# model.add(Dense(100))
# model.add(Dense(500))
# model.add(Dense(100))
# model.add(Dense(64,activation='relu'))
model.add(Dense(1))


# 3. compile, training
model.compile(
    optimizer='adam',
    loss='mae',
    metrics=['mse']
)

Early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=4, 
    verbose=2, 
    restore_best_weights=True
    )


hist=model.fit(
    x_train, y_train,
    batch_size=1,
    epochs=1500,
    validation_split=0.2,
    callbacks=[Early_stopping]
)


# 4. 평가 ,예측


loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)


def RMSE(y_test, y_predict):
    return(np.sqrt(mean_squared_error(y_test, y_predict)))


print('loss', loss)
print('RMSE', RMSE(y_test, y_predict))
print('r2', r2_score(y_test, y_predict))
print('hist : ',hist.history['val_loss'])
# r2 값 0.62 이상

"""
결과
loss [39.077388763427734, 39.077388763427734]
RMSE 47.94034590318054
r2 0.6345808155342813
"""


# 5. 시각화 (label, title, grid, figsize, color, marker, data, legend)

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
plt.title('diabets')
plt.legend(loc='upper right')
plt.show()








