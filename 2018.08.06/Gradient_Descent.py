import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
m, n = housing.data.shape
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)

n_epochs = 1000
learning_rate = 0.01
scaled_housing_data_plus_bias = np.c_[np.ones((m,1)), scaled_housing_data]
X = tf.constant(scaled_housing_data_plus_bias, dtype = tf.float32, name = "X")
y = tf.constant(housing.target.reshape(-1, 1), dtype = tf.float32, name = "y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name = "theta")
y_pred = tf.matmul(X, theta, name = "predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()

# 반복루프: 훈련단계를 반복해서 실행(n_ecpochs)
#-100번 반복마다 현재의 평균 제곱에러(mse)출력
#MSE는 매 반복에서 값이 줄어들어야 함

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("에포크 ", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()

print("best_theta:")
print(best_theta)