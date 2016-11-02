# -*- using: utf-8 -*-

from cache import save_path,save_path2, csv_path

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import pandas as pd
from sklearn.datasets import load_iris, load_digits, load_boston
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error, mean_absolute_error
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV
import numpy as np
from numpy import mean,var
from math import ceil, sqrt
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from datapre import balance
from utils import save_file, actual_filter, write_metrics

rng = np.random.RandomState(31337)


def _times(i):
    if i==1:
        return 0
    else:
        return 1

def _time(i):
    if i==0:
        return 0
    else:
        return 1

# 导入数据
data = pd.read_csv(csv_path)
# data = data[0:500]
# data,_ = train_test_split(data,train_size=0.2)
# data['coarticle_num'] = list(map(_times, data['coarticle_num'].values))
# data['colla_time'] = list(map(_time, data['colla_time'].values))
# print(len(data['coarticle_num'].values))


def xgboost_classifier(data, feature_cols, response_col, thresholds=[0.5]):
    # X = data[feature_cols][0:100].values
    # y = data[response_col][0:100].values
    # kf = KFold(y.shape[0], n_folds=10, shuffle=True, random_state=rng)
    # for train_index, test_index in kf:
    #     xgb_model = xgb.XGBClassifier().fit(X[train_index], y[train_index])
    #     predictions = xgb_model.predict(X[test_index])
    #     actuals = y[test_index]
    #     print(confusion_matrix(actuals, predictions))
    X = data[feature_cols].values
    y = data[response_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)
    xgb_model = xgb.XGBClassifier(learning_rate=0.1,
        n_estimators=100,
        max_depth=5,
        min_child_weight=1,
        gamma=0.2,
        subsample=0.8,
        colsample_bytree=0.7,
        objective='reg:linear',
        nthread=-1,
        scale_pos_weight=1,
        seed=27, silent=False).fit(X_train, y_train)
    # predictions = xgb_model.predict(X_train)
    predictions_proba = xgb_model.predict_proba(X_train)[:,1]
    precisions = []
    recalls = []
    f1s = []
    accuracys = []
    for threshold in thresholds:
        predictions = [ceil(i-threshold) for i in predictions_proba]
        actuals = y_train
        print(confusion_matrix(actuals, predictions, labels=[0,1]))
        accuracy = metrics.accuracy_score(actuals, predictions)
        accuracys.append(accuracy)
        print("Accuracy : %.4g" % accuracy)
        con = confusion_matrix(actuals, predictions)
        tp,fp,tn,fn = con[1][1],con[0][1],con[0][0],con[1][0]
        print("precision = ", tp/(tp+fp))
        precision = round(tp/(tp+fp),3)
        precisions.append(precision)
        print("recall = ", tp/(tp+fn))
        recall = round(tp/(tp+fn), 3)
        recalls.append(recall)
        print("f1 = ", 2*tp/(2*tp+fn+fp))
        f1 = round(2*tp/(2*tp+fn+fp),3)
        f1s.append(f1)
        # print(f1_score(actuals, predictions))
    print("threshold = ", thresholds)
    print("presions = ", precisions)
    print("recalls = ", recalls)
    print("f1s = ", f1s)
    print("accuracys = ", accuracys)


def xgboost_regression(data, feature_cols, response_col, row=1, save_path=save_path, train_size=0.8, mode='not3', e=1):

    X = data[feature_cols].values
    y = data[response_col].values
    # X,y = balance(X,y,ratio=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng, train_size=train_size)
    y_train = [i**(1/e) for i in y_train]
    xgb_model = xgb.XGBRegressor(learning_rate=0.1, max_depth=3, n_estimators=100).fit(X_train, y_train)
    predictions = xgb_model.predict(X_test)
    predictions = [i**e for i in predictions]
    actual = y_test

    if mode == '3':
        if response_col == 'colla_time':
            for i in range(1, 41):
                a, b = actual_filter(actual, predictions, i)
                write_metrics(a, b, save_path, i, 2)
        elif response_col == 'coarticle_num':
            for i in range(1, 101):
                a, b = actual_filter(actual, predictions, i)
                write_metrics(a, b, save_path, i, 2)

    else:
        mae = mean_absolute_error(actual, predictions)
        print('mae = %.3f' % mae)
        save_file(save_path + 'mae.xls', row, 2, '%.3f' % mae)

        mse = mean_squared_error(actual, predictions)
        print('mse = %.3f' % mse)
        save_file(save_path + 'mse.xls', row, 2, '%.3f' % mse)

        rmse = sqrt(mse / len(predictions))
        print('rmse = %.3f' % rmse)
        save_file(save_path + 'rmse.xls', row, 2, '%.3f' % rmse)

        pcc = pearsonr(actual, predictions)[0]
        print('pcc = %.3f' % pcc)
        save_file(save_path + 'pcc.xls', row, 2, '%.3f' % pcc)

        ccc = 2 * pcc * sqrt(var(actual)) * sqrt(var(predictions)) / (
            var(actual) + var(predictions) + (mean(actual) - mean(predictions)) ** 2)
        print('ccc = %.3f' % ccc)
        save_file(save_path + 'ccc.xls', row, 2, '%.3f' % ccc)


    # predictions_proba = xgb_model.predict_proba(X_train)[:, 1]
    # precisions = []
    # recalls = []
    # f1s = []
    # accuracys = []
    # for threshold in thresholds:
    #     predictions = [ceil(i - threshold) for i in predictions_proba]
    #     actuals = y_train
    #     print(confusion_matrix(actuals, predictions, labels=[0, 1]))
    #     accuracy = metrics.accuracy_score(actuals, predictions)
    #     accuracys.append(accuracy)
    #     print("Accuracy : %.4g" % accuracy)
    #     con = confusion_matrix(actuals, predictions)
    #     tp, fp, tn, fn = con[1][1], con[0][1], con[0][0], con[1][0]
    #     print("precision = ", tp / (tp + fp))
    #     precision = round(tp / (tp + fp), 3)
    #     precisions.append(precision)
    #     print("recall = ", tp / (tp + fn))
    #     recall = round(tp / (tp + fn), 3)
    #     recalls.append(recall)
    #     print("f1 = ", 2 * tp / (2 * tp + fn + fp))
    #     f1 = round(2 * tp / (2 * tp + fn + fp), 3)
    #     f1s.append(f1)
    #     # print(f1_score(actuals, predictions))
    # print("threshold = ", thresholds)
    # print("presions = ", precisions)
    # print("recalls = ", recalls)
    # print("f1s = ", f1s)
    # print("accuracys = ", accuracys)


def xgb_start(response_col='colla_time', save_path=save_path):
    # 学术年龄
    print('属性是学术年龄和属性是除了学术年龄所有的')
    feature_cols = ['scientific_age1', 'scientific_age2']
    l = xgboost_regression(data, feature_cols, response_col, 1, save_path)
    # print(l)
    # 除了学术年龄
    feature_cols = ['article_num1', 'article_num2', 'common_neighbors_num',
                    'shortest_path_length', 'degree1', 'degree2']
    l = xgboost_regression(data, feature_cols, response_col, 2, save_path)
    # print(l)
    # 文章数
    print('文章数和除了文章数')
    feature_cols = ['article_num1', 'article_num2']
    l = xgboost_regression(data, feature_cols, response_col, 3, save_path)
    # print(l)
    feature_cols = ['scientific_age1', 'scientific_age2', 'common_neighbors_num',
                    'shortest_path_length', 'degree1', 'degree2']
    l = xgboost_regression(data, feature_cols, response_col, 4, save_path)
    # print(l)
    # 共同邻居
    print('共同邻居和除了共同邻居')
    feature_cols = ['common_neighbors_num']
    l = xgboost_regression(data, feature_cols, response_col, 5, save_path)
    # print(l)
    feature_cols = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2',
                    'shortest_path_length', 'degree1', 'degree2']
    l = xgboost_regression(data, feature_cols, response_col, 6, save_path)
    # print(l)
    # 最短路径
    print('最短路径和除了最短路径')
    feature_cols = ['shortest_path_length']
    l = xgboost_regression(data, feature_cols, response_col, 7, save_path)
    # print(l)
    feature_cols = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2', 'common_neighbors_num',
                    'degree1', 'degree2']
    l = xgboost_regression(data, feature_cols, response_col, 8, save_path)
    # print(l)
    # 度
    print('度和除了度')
    feature_cols = ['degree1', 'degree2']
    l = xgboost_regression(data, feature_cols, response_col, 9, save_path)
    # print(l)
    feature_cols = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2', 'common_neighbors_num',
                    'shortest_path_length']
    l = xgboost_regression(data, feature_cols, response_col, 10, save_path)
    # print(l)

    # 都有
    print("所有的属性")
    feature_cols = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2', 'common_neighbors_num',
                    'shortest_path_length', 'degree1', 'degree2']
    l = xgboost_regression(data, feature_cols, response_col, 11, save_path)


def xgb_start_trainsize(response_col='colla_time', save_path=save_path):
    feature_cols = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2', 'common_neighbors_num',
                    'shortest_path_length', 'degree1', 'degree2']
    xgboost_regression(data, feature_cols, response_col,1, save_path, 0.2)
    xgboost_regression(data, feature_cols, response_col,2, save_path, 0.3)
    xgboost_regression(data, feature_cols, response_col,3, save_path, 0.4)
    xgboost_regression(data, feature_cols, response_col,4, save_path, 0.5)
    xgboost_regression(data, feature_cols, response_col,5, save_path, 0.6)
    xgboost_regression(data, feature_cols, response_col,6, save_path, 0.7)
    xgboost_regression(data, feature_cols, response_col,7, save_path, 0.8)
    xgboost_regression(data, feature_cols, response_col,8, save_path, 0.9)

    # xgboost_regression(data, feature_cols, response_col, 1, save_path, 0.001)
    # xgboost_regression(data, feature_cols, response_col, 2, save_path, 0.004)
    # xgboost_regression(data, feature_cols, response_col, 3, save_path, 0.007)
    # xgboost_regression(data, feature_cols, response_col, 4, save_path, 0.01)
    # xgboost_regression(data, feature_cols, response_col, 5, save_path, 0.1)
    # xgboost_regression(data, feature_cols, response_col, 6, save_path, 0.7)
    # xgboost_regression(data, feature_cols, response_col, 7, save_path, 0.8)
    # xgboost_regression(data, feature_cols, response_col, 8, save_path, 0.9)


def xgb_start_3(response_col='colla_time', save_path=save_path):
    feature_cols = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2', 'common_neighbors_num',
                    'shortest_path_length', 'degree1', 'degree2']
    xgboost_regression(data=data, feature_cols=feature_cols,response_col=response_col,save_path=save_path,mode='3')


def modelfit(alg, dtrain, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()


if __name__ == '__main__':
    predictors = ['scientific_age1','scientific_age2','article_num1','article_num2','common_neighbors_num',
                    'shortest_path_length','degree1','degree2']
    # predictors = ['shortest_path_length']
    # target = 'colla_time'
    # target = 'coarticle_num'
    # xgboost_classifier(data, predictors, target, np.arange(0.1,1,0.1))
    # xgboost_regression(data, predictors, target)
    # 1
    xgb_start()
    xgb_start('coarticle_num', save_path2)
    # 2
    # xgb_start_trainsize()
    # xgb_start_trainsize(response_col='coarticle_num', save_path=save_path2)
    # 3
    # xgb_start_3()
    # xgb_start_3(response_col='coarticle_num', save_path=save_path2)




# 调参
# step1
# predictors = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2', 'common_neighbors_num',
#                   'shortest_path_length', 'degree1', 'degree2']
# # # response_col = 'colla_time'
# target = 'coarticle_num'

# step2
# param_test1 = {
#  'max_depth': [4,5,6],
#  'min_child_weight': [1,2]
# }
# gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=5,
#  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27, silent=False),
#  param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch1.fit(data[predictors],data[target])
# print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_, sep='\n')

# step3
# param_test3 = {
#  'gamma': [i/10.0 for i in range(0,5)]
# }
# gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
#  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27,silent=False),
#  param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch3.fit(data[predictors],data[target])
# print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_, sep='\n')

# step4
# param_test4 = {
#  'subsample':[i/10.0 for i in range(6,10)],
#  'colsample_bytree':[i/10.0 for i in range(6,10)]
# }
# gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
#  min_child_weight=1, gamma=0.2, subsample=0.8, colsample_bytree=0.8,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27,silent=False),
#  param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch4.fit(data[predictors],data[target])
# print(gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_,sep='\n')

# step5
# param_test6 = {
#  # 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
#     'reg_alpha': [0.7, 0.8, 0.9, 1]
# }
# gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=4,
#  min_child_weight=1, gamma=0.2, subsample=0.8, colsample_bytree=0.7,
#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
#  param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
# gsearch6.fit(data[predictors],data[target])
# print(gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_, sep='\n')

# xgb1 = XGBClassifier(
#     learning_rate=0.1,
#     n_estimators=100,
#     max_depth=5,
#     min_child_weight=1,
#     gamma=0,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective='binary:logistic',
#     nthread=-1,
#     scale_pos_weight=1,
#     seed=27, silent=False)
# xgb2 = XGBClassifier(
#     learning_rate=0.1,
#     n_estimators=100,
#     max_depth=5,
#     min_child_weight=1,
#     gamma=0.2,
#     subsample=0.8,
#     colsample_bytree=0.7,
#     objective='reg:linear',
#     booster:'gblinear',
#     nthread=-1,
#     scale_pos_weight=1,
#     seed=27, silent=False)
# modelfit(xgb2, data, predictors, target)