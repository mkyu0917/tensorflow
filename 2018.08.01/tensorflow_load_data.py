import tensorflow as tf

filenanme_queue = tf.train.string_input_producer(['test-score.csv'], shuffle= False, name = 'filenanme_queue')
#suffle = True 데이터를 각각 컬럼값에서 섞어서 가져옴
reader = tf.TextLineReader()
key, value = reader.read(filenanme_queue)

# decoded result

record_defaults = [[0.],[0.],[0.],[0.]]
xy = tf.decode_csv(value, record_defaults = record_defaults)
 #collect batches of csv in

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1],xy[-1:0]], batch_size=10)

 #placeholders for a tensor that will be always fed

X = tf.placeholder(tf.float32, shape = [None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([3,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X,W) + b

# SImplified cost / loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

#Launch the graph in a session.
sess = tf.Session()

#Initializes global varianble in the graph.
sess.run(tf.global_variables_initializer())
#Start populationg the flilname queue.
#-------------------------------------------------------------------
coord = tf.train.Coordinator() #기본적으로 외워야하는 코드
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#-------------------------------------------------------------------
for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch,train_y_batch])
    cost_val, hy_val,_ =sess.run(
    [cost, hypothesis, train], feed_dict={X:x_batch, Y:y_batch})
    if step % 10 == 0:
     print(step,"Cost:", cost_val, "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)

#Ask my score

print("Your score will be",
      sess.run(hypothesis, feed_dict= {X: [[100,70,101]]}))
print("Other scores will be",
      sess.run(hypothesis, feed_dict={X:[[60, 70, 110],[90, 100, 80]]}))