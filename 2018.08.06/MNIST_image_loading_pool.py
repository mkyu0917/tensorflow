from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
#Check out https://www.tensorflow.org/get_started/mnist/beginners for
#more information about the mnist dataset

img = mnist.train.images[0].reshape(28, 28)
sess = tf.InteractiveSession()
img = img.reshape(-1, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01)) # 3X3 1개 color 5개 filter
conv2d = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding='SAME') # 2칸씩 이미지 이동(14X14)
pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding = 'SAME')
print(pool)
sess.run(tf.global_variables_initializer())
pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(7, 7), cmap='gray')
plt.show()