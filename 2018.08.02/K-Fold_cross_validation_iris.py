import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import scipy

def import_data():
    #import total dataset
    data = pd.read_csv('iris_data.csv')
    #get a list of column names
    headers = list(data.columns.values)
    #separate into independent and dependent variables
    x = data[headers[:4]]
    print(x)
    y = data[headers[-1:]].values.ravel()
    print(y)
    return x,y


if __name__=='__main__':
    # get training and testing sets
    x, y = import_data()
    # set to 10 folds
    skf=StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(x,y):
        # specific ".loc" syntax for working with dataframes
        x_train, x_test = x.loc[train_index], x.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #create and fit classifier
        classifier = GaussianNB()
        classifier.fit(x_train,y_train)
        #classify our test variables
        predictions = classifier.predict(x_test)
        #save and print accuracy
        accuracy = metrics.accuracy_score(y_test, predictions)
        print("Accuracy:" + accuracy.__str__())