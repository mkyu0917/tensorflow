import tensorflow as tf

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,7,7]]

y_data = [[0,0,1], # one-hot인코딩 (A,B,C)
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]



X = tf.placeholder("float",[None, 4])
Y = tf.placeholder("float",[None, 3])
nb_classes = 3 #분류를 3개로

W = tf.Variable(tf.random_normal([4, nb_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')

#tf.nn.softmax computes softmax activations
# sotfmax = exp(Logits) / reduce_sum(exp(Logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X,W)+b) #oftmax(소프트맥스)는 입력받은 값을 출력으로 0~1사이의 값으로 모두 정규화하며 출력 값들의 총합은 항상 1이 되는 특성을 가진 함수
#행렬간 곱셈을 위해서는 ‘*‘가 아닌 matmul 함수를 이용하여 곱
# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis),axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y:y_data}))


#Testing & One-hot encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.arg_max(a,1))) # argmax 가장 높은값을 알려줌,

    all = sess.run(hypothesis, feed_dict={X:[[1, 11, 7, 9],
                                             [1, 3, 4, 3 ],
                                             [1, 1, 0, 1]]})
    print(all, sess.run(tf.arg_max(all, 1)))