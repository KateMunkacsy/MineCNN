'''
Trying to use gpu to run a Keras model
    Engine details- 
      Image: Complete Geo Engine 1
      Kernel: Python 3
      Profile: 
        4 vCPU/32 GiB Memory
        1 GPU
'''


import keras              # version 2.4.3
import tensorflow as tf   # version 2.2.0


config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.compat.v1.Session(config=config)