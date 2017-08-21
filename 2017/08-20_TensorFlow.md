# ___2017 - 08 - 20 TensorFlow___

# 目录

# Hello World
```python
import tensorflow as tf
hello = tf.constant('Hello TensorFlow')
sess = tf.Session()
sess.run(hello)
Out[5]: b'Hello TensorFlow'

a = tf.constant(10)
b = tf.constant(32)
sess.run(a+b)
Out[9]: 42
```
