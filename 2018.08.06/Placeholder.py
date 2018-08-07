import tensorflow as tf

# 실제로 아무 계산도 하지 않는 특수한 노드, 실행시 데이터를 출력
# 훈련을 하는 동안 tensor에 훈련을 데이터를 전달하기 위해
# (실행시 placeholder에 값을 지정하지 않으면 예외 발생)
A = tf.placeholder(tf.float32, shape=(None, 3)) #껍데기만 만들어줌
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]}) #A라는 값에 1,2,3이라는 행을 넣음
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6],[7, 8 ,9]]})
print(B_val_1)
print(B_val_2)