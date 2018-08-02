


###선형회귀 : linear regression###
##모델에서 가장 중요한 cost값을 찾아준다.

import tensorflow as tf

tf.set_random_seed(777) # for reproducibility

#tf Graph Input
X = [1,2,3]
Y = [1,2,3]

#Set Wrong model weights
W = tf.Variable(-3.0) #5일경우

#Linear model
hypothesis = X * W

#cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Minimize Gradient Descent Magic
#모델에서 가장 중요한 cost값을 찾아준다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

#Launch the graph in a session
sess = tf.Session()

#Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())


for step in range(100):
    print(step,sess.run(W))
    sess.run(train)