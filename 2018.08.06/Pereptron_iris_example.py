#Load required libraries

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Load the iris dataset
iris = datasets.load_iris()

#Create our X and y data
X = iris.data
y = iris.target

# View the first five ovservations of our y data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#Train the scaler, which standrizes all the features to have mean = 0 and unit variance
sc = StandardScaler()
sc.fit(X_train)
#Apply the scaler to the X training data
X_train_std = sc.transform(X_train)
#Apply the scaler to the X test data
X_test_std = sc.transform(X_test)

#Create a perceptron object with the parameters
#40 iterations (epochs) over the data, and a learning rate of 0.1

ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)

#Train the perceptron
ppn.fit(X_train_std, y_train)

#apply the trained perceptron on the X data to make perdicts the y test data
y_pred = ppn.predict(X_test_std)

#View the predicted y test data
print(y_pred)
#View the true y test data
print(y_test)
#view the accuracy of the model, which is : 1 - (observations predicted wrong / total observations)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))