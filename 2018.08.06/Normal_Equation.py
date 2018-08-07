import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data] #편향에 대한 입력 특성 (X0=1)을 추가
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
XT = tf.transpose(X) #전차 행렬
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)),XT),y) #inverse 역행렬
print(X)
print(y)
print(XT)
print(theta)
with tf.Session() as sess:
    theta_value = theta.eval()
    print(theta_value)

#Tenseor("x:0", shape(20640, 9), dtype=float32)
#Tenseor("x:0", shape(20640, 9), dtype=float32)
#Tenseor("Transpose:0", shape(9, 20640), dtype=float32)
#Tenseor("MatMul_2:0", shape(9,1), dtype=float32)

