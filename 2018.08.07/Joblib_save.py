from sklearn import datasets
from sklearn.externals import joblib
import pickle

iris = datasets.load_iris()
X=iris.data
y=iris.target
from sklearn.model_selection import train_test_split
# 데이터를 트레이닝셋과 테스트 셋으로 나눔, 7:3
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.3, random_state=0)

pickle.dump(train_X,open('train_X.pkl','wb'))
pickle.dump(train_X,open('test_X.pkl','wb'))
pickle.dump(train_X,open('train_Y.pkl','wb'))
pickle.dump(train_X,open('test_Y.pkl','wb'))

from sklearn.preprocessing import MinMaxScaler
#나눈 값의
scaler = MinMaxScaler() # 0~1사이 값을 나오게 하여 학습결과를 좋게하는 것이다.
scaler.fit(train_X)
train_x_scaled = scaler.transform(train_X) # 실제 자료를 변환
print(train_x_scaled[:5])

file_name = 'scaler_01.pkl'
joblib.dump(scaler,file_name)