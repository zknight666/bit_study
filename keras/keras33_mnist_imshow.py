import numpy as np
from tensorflow.keras.datasets import mnist






(x_train,y_train), (x_test,y_test) = mnist.load_data()


print(x_train.shape) # (60000, 28, 28) 6만장, 28x28 사이즈의 흑백
print(y_train.shape) # (60000,) 

# input shape 넣기 위해 reshape 함 (60000,28,28,1)이나 60000,28,28이나 같아서

print(x_test.shape) # (10000, 28, 28)
print(y_test.shape) # (10000,)


print(x_train[0]) # 28x28 
print(y_train[0]) # 5

import matplotlib.pyplot as plt
plt.imshow(x_train[1000],'gray')
plt.show()









