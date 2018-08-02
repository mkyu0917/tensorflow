import numpy as np

xy = np.loadtxt('test-score.csv',delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1] # 끝 뺴고 나머지것 독립변수로 쓴다.
#print('xdata=',x_data)
y_data = xy [:, [-1]]
#print('ydata=',y_data)

#print(x_data.shape, x_data, len(x_data))
#print(y_data.shape, y_data, len(x_data))
#
# print(xy[2:4])
#
# print(xy[2:4])
# print(xy[2:])
# print(xy[:2])
# print(xy[:-1])
# xy[2:4] = [8,9]
#print(xy)