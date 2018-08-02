import tensorflow as tf

x_data = {[73, 80, 75],
          [93, 88, 93],
          [89, 91, 90],
          [96, 98, 100],
          [73, 66, 70]}
y_data = {[152.],
          [185.],
          [180.],
          [196.],
          [142.]}

X = tf.placeholder(tf.float32, shape = [None, 3])
Y = tf.