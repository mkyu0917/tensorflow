from sklearn import datasets
from sklearn.externals import joblib
import pickle
from sklearn.preprocessing import MinMaxScaler

train_X = pickle.load(open('train_X.pkl','rb'))
test_X = pickle.load(open('test_X.pkl','rb'))
train_Y = pickle.load(open('train_Y.pkl','rb'))
train_Y = pickle.load(open('test_Y.pkl','rb'))

file_name = 'scaler_01.pkl'
scaler = joblib.load(file_name)
train_x_scaled = scaler.transform(train_X)
print(train_x_scaled[:5])
