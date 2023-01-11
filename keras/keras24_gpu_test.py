import numpy as np
import tensorflow as tf



print(tf.__version__) # 2.7.4



gpus = tf.config.experimental.list_physical_devices('GPU')

print(gpus)

if(gpus):
    print("돌았음")
else:
    print("안돔")








