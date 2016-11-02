from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import OneSidedSelection
from utils import actual_filter
from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def balance(x, y, ratio=0.1, min=0, max=39, min_len=100):
    x = list(x)
    y = list(y)
    # rus = RandomUnderSampler(random_state=42, ratio=ratio)
    # X_res, y_res = rus.fit_sample(X, y)

    # cc = ClusterCentroids(random_state=42)
    # X_res, y_res = cc.fit_sample(X, y)

    # oss = OneSidedSelection(random_state=42)
    # X_res, y_res = oss.fit_sample(X, y)
    y_res = []
    x_res = []
    for i in range(min, max+1):
        y_t, x_t = actual_filter(y, x, i)
        if min_len/ratio < len(y_t):
            x_t, X_test, y_t, y_test = train_test_split(x_t, y_t, random_state=42, train_size=(min_len/ratio)/len(y_t))
        if len(y_t) > 0:
            y_res += y_t
            x_res += x_t
    # print(y_res)
    return x_res, y_res


def mean_normalization(X, min=-1, max=1):
    min_max_scaler = MinMaxScaler(feature_range=(min,max))
    X_minmax = min_max_scaler.fit_transform(X)
    return X_minmax

def test():
    X = np.array([[1,2,3],[-1,5,6],[0,3,1]])
    t = mean_normalization(X,0,1)
    print(X)
    print(t)

if __name__ == '__main__':
    test()