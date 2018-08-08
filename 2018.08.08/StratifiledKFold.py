import numpy as np
seed = 0
np.random.seed(seed)

# 원소가 9개인 numpy 배열을 생성한다.
# Y값은 0이 3개 , 1은 6개로 비율은 1:2이다 (불균형데이터)
X = np.array([-5,-3,-1,1,3,5,7,9,11])
Y = np.array([0,0,0,1,1,1,1,1,1])
splits =3

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

print(kfold)
print("=" * 100)
for train_index, test_index in kfold.split(X,Y):
    print("train Index:", train_index)
    print("test Index:", train_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    print("-"*100)

# StratifiedShuffleSplit
# Stratified하게 트레이닝셋과 테스트셋으로 나눈다.
# test에 선택된 인덱스 겹쳐도 되며 splits개수만큼 추출한다.]

from sklearn.model_selection import StratifiedShuffleSplit

shufflesplit = StratifiedShuffleSplit(n_splits=splits, random_state=seed, test_size=0.3)

print(shufflesplit)
print("="*100)
for train_index, test_index in shufflesplit.split(X, Y):
    print("train Index:", train_index)
    print("test index:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    print("-" * 100)

from sklearn.model_selection import train_test_split

train_index, test_index = train_test_split(np.array(range(X.shape[0])), shuffle=True, stratify=Y,
                                           test_size=0.3, random_state=seed)

print("train Index:", train_index)
print("test index:", test_index)
X_train, X_test = X[train_index], X[test_index]
Y_train, Y_test = Y[train_index], Y[test_index]
print("-"*100)