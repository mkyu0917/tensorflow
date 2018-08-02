
import  tensorflow as tf
import matplotlib.pyplot as plt
X = [1,2,3]
Y = [1,2,3]

W= tf.placeholder(tf.float32)
print(W)
# Our hypothesis for linear model X * W

hypothesis = X * W

#cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Launch the graph in session
sess = tf.Session()
#Variables for plotting cost function
W_history = []
cost_history = []
for i in range(-30,50):
    curr_W = 0.1 * i
    curr_cost = sess.run(cost, feed_dict={W:curr_W})
    W_history.append(curr_W)
    cost_history.append(curr_cost)

# show the cost function
plt.plot(W_history, cost_history)
plt.show()