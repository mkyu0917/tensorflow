import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

sentence = ("if you wnat to build a ship, don't drum up people together to"
            "collect wood and don't assign them tasks and work, but rather"
            "teach them to long for the endless Immensity of the sea")

char_set = list(set(sentence)) #중복없는 알파벳 집합 만들고 리스트 변환print(char_set)
char_dic = {w: i for i, w in enumerate(char_set)} # 알파벳을 key, 인덱스를 value로 하는 딕셔너릭 생성
data_dim = len(char_set)
hidden_size = len(char_set) # 각 셀의 출력크기
num_classes = len(char_set) # 분류 총수
sequence_length = 10 # any arbitrary number # 1개의 시퀸스의 길이 ( 시계열 데이터의 입력갯수)
learning_rate = 0.1
dataX = [] #입력 시퀸스를 저장하기 위한 배열
dataY = [] #입력 시퀸스를 저장하기 위한 배열

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length]
    print( i ,x_str, '->', y_str)

    x= [char_dic[c] for c in x_str] # x str to index
    y = [char_dic[c] for c in y_str]  # x str to index

    dataX.append(x) # array만듬
    dataY.append(y)

batch_size = len(dataX) #전체 169 알아서 잘라줌
X = tf.placeholder(tf.int32, [None, sequence_length]) #Xdata
Y = tf.placeholder(tf.int32, [None, sequence_length]) #Ydata

#One-hot encoding
X_one_hot = tf.one_hot(X, num_classes) #전체분류개수 1개 차원 추가
print(X_one_hot) #check out the shape
# Make a lstm cell with hidden_size (each unit output vacter)
def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell

multi_cells = rnn.MultiRNNCell([lstm_cell() for _in range(2)], state_is_tuple=True)
# outputs : unfolding size X hidden size, state = hidden size

