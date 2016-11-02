
from cache import save_path, save_path2, csv_path

# feature_all = ['scientific_age1', 'scientific_age2', article_num1, article_num2,
# common_neighbors_num, shortest_path_length, degree1, degree2,scientific_age_to2016_1, scientific_age_to2016_2,
# article_num_to2016_1, article_num_to2016_2]
# feature_cols = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2', 'common_neighbors_num',
#                 'shortest_path_length', 'degree1', 'degree2']

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.stats import pearsonr
from numpy import mean, var
from math import ceil, sqrt
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error, mean_absolute_error
import numpy as np
import seaborn as sns
from collections import Counter
from datapre import balance
from utils import save_file, actual_filter, write_metrics
from datapre import mean_normalization
data = pd.read_csv(csv_path)


def linear_regression(data, feature_cols, response_col, row=1, save_path=save_path, train_size=0.8, mode='not3', e=1, b=0):
    X = data[feature_cols].values
    mean_normalization(X)
    y_time = data[response_col].values

    # print('Resampled dataset shape {}'.format(Counter(y_time)))
    # if response_col=='colla_time':
    #     t_max=40
    # else:
    #     t_max = 100
    # X, y_time = balance(X, y_time, ratio=0.01, min=0, max=39, min_len=156)
    # print('Resampled dataset shape {}'.format(Counter(y_time)))

    X_train, X_test, y_train, y_test = train_test_split(X, y_time, random_state=1, train_size=train_size)
    y_train = [i**(1/e)+b for i in y_train]
    linreg = LinearRegression(n_jobs=4)
    linreg.fit(X_train, y_train)

    predictions = linreg.predict(X_test)
    predictions = [(i-b)**e for i in predictions]
    actual = y_test

    if mode == '3':
        if response_col == 'colla_time':
            for i in range(1, 41):
                a, b = actual_filter(actual, predictions, i)
                write_metrics(a, b, save_path, i, 1)
        elif response_col == 'coarticle_num':
            for i in range(1, 101):
                a, b = actual_filter(actual, predictions, i)
                write_metrics(a, b, save_path, i, 1)

    else:
        mae = mean_absolute_error(actual, predictions)
        print('mae = %.3f' % mae)
        save_file(save_path+'mae.xls', row, 1, '%.3f'%mae)

        mse = mean_squared_error(actual, predictions)
        print('mse = %.3f' % mse)
        save_file(save_path + 'mse.xls', row, 1, '%.3f' % mse)

        rmse = sqrt(mse / len(predictions))
        print('rmse = %.3f' % rmse)
        save_file(save_path + 'rmse.xls', row, 1, '%.3f' % rmse)

        pcc = pearsonr(actual, predictions)[0]
        print('pcc = %.3f' % pcc)
        save_file(save_path + 'pcc.xls', row, 1, '%.3f' %pcc)

        ccc = 2 * pcc * sqrt(var(actual)) * sqrt(var(predictions)) / (var(actual) + var(predictions) + (mean(actual) - mean(predictions)) ** 2)
        print('ccc = %.3f' % ccc)
        save_file(save_path + 'ccc.xls', row, 1, '%.3f' %ccc)


def saveplot(data, feature_cols, response_col, fig_name):
    fig = sns.pairplot(data, x_vars=feature_cols, y_vars=response_col, markers='.', kind='scatter')
    fig.savefig('./fig/' + fig_name + '.pdf', dpi='figure')
    print(fig_name + ' saved successfully.')


def linreg_start(response_col='colla_time', save_path=save_path):

    # 学术年龄
    print('属性是学术年龄和属性是除了学术年龄所有的')
    feature_cols = ['scientific_age1', 'scientific_age2']
    l = linear_regression(data, feature_cols, response_col,1, save_path)
    # print(l)
    # 除了学术年龄
    feature_cols = ['article_num1', 'article_num2', 'common_neighbors_num',
                    'shortest_path_length', 'degree1', 'degree2']
    l = linear_regression(data, feature_cols, response_col,2, save_path)
    # print(l)
    # 文章数
    print('文章数和除了文章数')
    feature_cols = ['article_num1', 'article_num2']
    l = linear_regression(data, feature_cols, response_col,3, save_path)
    # print(l)
    feature_cols = ['scientific_age1', 'scientific_age2', 'common_neighbors_num',
                    'shortest_path_length', 'degree1', 'degree2']
    l = linear_regression(data, feature_cols, response_col,4, save_path)
    # print(l)
    # 共同邻居
    print('共同邻居和除了共同邻居')
    feature_cols = ['common_neighbors_num']
    l = linear_regression(data, feature_cols, response_col,5, save_path)
    # print(l)
    feature_cols = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2',
                    'shortest_path_length', 'degree1', 'degree2']
    l = linear_regression(data, feature_cols, response_col,6, save_path)
    # print(l)
    # 最短路径
    print('最短路径和除了最短路径')
    feature_cols = ['shortest_path_length']
    l = linear_regression(data, feature_cols, response_col,7, save_path)
    # print(l)
    feature_cols = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2', 'common_neighbors_num',
                    'degree1', 'degree2']
    l = linear_regression(data, feature_cols, response_col,8, save_path)
    # print(l)
    # 度
    print('度和除了度')
    feature_cols = ['degree1', 'degree2']
    l = linear_regression(data, feature_cols, response_col,9, save_path)
    # print(l)
    feature_cols = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2', 'common_neighbors_num',
                    'shortest_path_length']
    l = linear_regression(data, feature_cols, response_col,10, save_path)
    # print(l)

    # 都有
    print("所有的属性")
    feature_cols = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2', 'common_neighbors_num',
                    'shortest_path_length', 'degree1', 'degree2']
    l = linear_regression(data, feature_cols, response_col,11, save_path)
    # print(l)


    # print('合作次数：')
    # response_col = 'coarticle_num'
    # # 学术年龄
    # print('学术年龄和除了学术年龄')
    # feature_cols = ['scientific_age1', 'scientific_age2']
    # l = linear_regression(data, feature_cols, response_col)
    # # print(l)
    # # 除了学术年龄
    # feature_cols = ['article_num1', 'article_num2', 'common_neighbors_num',
    #                 'shortest_path_length', 'degree1', 'degree2']
    # l = linear_regression(data, feature_cols, response_col)
    # # print(l)
    # # 文章数
    # print('文章数和除了文章数')
    # feature_cols = ['article_num1', 'article_num2']
    # l = linear_regression(data, feature_cols, response_col)
    # # print(l)
    # feature_cols = ['scientific_age1', 'scientific_age2', 'common_neighbors_num',
    #                 'shortest_path_length', 'degree1', 'degree2']
    # l = linear_regression(data, feature_cols, response_col)
    # # print(l)
    # # 共同邻居
    # print('共同邻居和除了共同邻居')
    # feature_cols = ['common_neighbors_num']
    # l = linear_regression(data, feature_cols, response_col)
    # # print(l)
    # feature_cols = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2',
    #                 'shortest_path_length', 'degree1', 'degree2']
    # l = linear_regression(data, feature_cols, response_col)
    # # print(l)
    # # 最短路径
    # print('最短路径和除了最短路径')
    # feature_cols = ['shortest_path_length']
    # l = linear_regression(data, feature_cols, response_col)
    # # print(l)
    # feature_cols = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2', 'common_neighbors_num',
    #                 'degree1', 'degree2']
    # l = linear_regression(data, feature_cols, response_col)
    # # print(l)
    # # 度
    # print('度和除了度')
    # feature_cols = ['degree1', 'degree2']
    # l = linear_regression(data, feature_cols, response_col)
    # # print(l)
    # feature_cols = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2', 'common_neighbors_num',
    #                 'shortest_path_length']
    # l = linear_regression(data, feature_cols, response_col)
    # # print(l)
    #
    # print("所有的属性")
    # feature_cols = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2', 'common_neighbors_num',
    #                 'shortest_path_length', 'degree1', 'degree2']
    # l = linear_regression(data, feature_cols, response_col)
    # # print(l)


def linreg_start_trainsize(response_col='colla_time', save_path=save_path):
    feature_cols = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2', 'common_neighbors_num',
                    'shortest_path_length', 'degree1', 'degree2']
    # linear_regression(data, feature_cols, response_col,1, save_path, 0.000001)
    # linear_regression(data, feature_cols, response_col,2, save_path, 0.000004)
    # linear_regression(data, feature_cols, response_col,3, save_path, 0.000007)
    # linear_regression(data, feature_cols, response_col,4, save_path, 0.00001)
    # linear_regression(data, feature_cols, response_col,5, save_path, 0.0001)
    # linear_regression(data, feature_cols, response_col,6, save_path, 0.001)
    # linear_regression(data, feature_cols, response_col,7, save_path, 0.01)
    # linear_regression(data, feature_cols, response_col,8, save_path, 0.1)

    linear_regression(data, feature_cols, response_col, 1, save_path, 0.2)
    linear_regression(data, feature_cols, response_col, 2, save_path, 0.3)
    linear_regression(data, feature_cols, response_col, 3, save_path, 0.4)
    linear_regression(data, feature_cols, response_col, 4, save_path, 0.5)
    linear_regression(data, feature_cols, response_col, 5, save_path, 0.6)
    linear_regression(data, feature_cols, response_col, 6, save_path, 0.7)
    linear_regression(data, feature_cols, response_col, 7, save_path, 0.8)
    linear_regression(data, feature_cols, response_col, 8, save_path, 0.9)

def linreg_start_3(response_col='colla_time', save_path=save_path):
    # feature_cols = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2', 'common_neighbors_num',
    #                 'shortest_path_length', 'degree1', 'degree2']
    # feature_cols = ['scientific_age1', 'scientific_age2']
    feature_cols = ['common_neighbors_num']
    linear_regression(data=data, feature_cols=feature_cols,response_col=response_col,save_path=save_path,mode='3')

def plot_test():
    response_col = 'colla_time'
    feature_cols = ['common_neighbors_num']
    saveplot(data, feature_cols, response_col, 'test')

def plot_start():
    print('合作时长：')
    response_col = 'colla_time'
    # 学术年龄
    print('学术年龄')
    feature_cols = ['scientific_age1', 'scientific_age2']
    saveplot(data, feature_cols, response_col, 'scientific_age')

    # 文章数
    print('文章数')
    feature_cols = ['article_num1', 'article_num2']
    saveplot(data, feature_cols, response_col, 'article_num')
    # 共同邻居
    print('共同邻居')
    feature_cols = ['common_neighbors_num']
    saveplot(data, feature_cols, response_col, 'common_neighbors_num')
    # 最短路径
    print('最短路径')
    feature_cols = ['shortest_path_length']
    saveplot(data, feature_cols, response_col, 'shortest_path_length')
    # 度
    print('度')
    feature_cols = ['degree1', 'degree2']
    saveplot(data, feature_cols, response_col, 'degree')

    print('合作次数：')
    response_col = 'coarticle_num'
    # 学术年龄
    print('学术年龄')
    feature_cols = ['scientific_age1', 'scientific_age2']
    saveplot(data, feature_cols, response_col, 'scientific_age')

    # 文章数
    print('文章数')
    feature_cols = ['article_num1', 'article_num2']
    saveplot(data, feature_cols, response_col, 'article_num')
    # 共同邻居
    print('共同邻居')
    feature_cols = ['common_neighbors_num']
    saveplot(data, feature_cols, response_col, 'common_neighbors_num')
    # 最短路径
    print('最短路径')
    feature_cols = ['shortest_path_length']
    saveplot(data, feature_cols, response_col, 'shortest_path_length')
    # 度
    print('度')
    feature_cols = ['degree1', 'degree2']
    saveplot(data, feature_cols, response_col, 'degree')


if __name__ == '__main__':
    # plot_start()
    # linreg_start()
    # predictors = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2', 'common_neighbors_num',
    #               'shortest_path_length', 'degree1', 'degree2']
    # target = 'colla_time'
    # target = 'coarticle_num'
    # linear_regression(data,predictors,target)
    # 1
    # linreg_start()
    # linreg_start(response_col='coarticle_num', save_path=save_path2)
    # 2
    # linreg_start_trainsize()
    # linreg_start_trainsize(response_col='coarticle_num', save_path=save_path2)
    # 3
    linreg_start_3()
    linreg_start_3(response_col='coarticle_num', save_path=save_path2)

# # 导入数据
# data = pd.read_csv(csv_path)
#
# # 特征
# feature_cols = ['common_neighbors_num']
# X = data[feature_cols]
# # 响应
# y_time = data['colla_time']
# # 训练集 测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y_time, random_state=1)
#
# # 线性回归模型 训练
# linreg = LinearRegression()
# linreg.fit(X_train, y_train)
#
# # 模型
# zip(feature_cols, linreg.coef_)
# print(linreg.intercept_）
# linreg.coef_
#
# # 评测
# y_pred = linreg.predict(X_test)
# np.sqrt(metrics.mean_absolute_error(y_test, y_pred))