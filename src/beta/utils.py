import xlrd
from xlutils.copy import copy
from scipy.stats import pearsonr
from numpy import mean, var
from math import ceil, sqrt
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error, mean_absolute_error


def save_file(file,row,col,content):
    """
    mae等度量保存到excel
    Args:
        file:
        row:
        col:
        content:

    Returns:

    """
    rd = xlrd.open_workbook(file)
    wd = copy(rd)
    ws = wd.get_sheet(0)
    ws.write(row, col, content)
    wd.save(file)


def actual_filter(a, b, a_value):
    """
    第三种图的筛选函数
    Args:
        a:
        b:
        a_value: 要筛选的值

    Returns:

    """
    actual = []
    predictions = []
    for i in range(len(a)):
        if a[i] == a_value:
            actual.append(a_value)
            predictions.append(b[i])
    return actual, predictions


def write_metrics(actual, predictions, save_path, row, col):
    if len(actual)>0:
        mae = mean_absolute_error(actual, predictions)
        print('mae = %.3f' % mae)
        save_file(save_path + 'mae.xls', row, col, '%.3f' % mae)

        mse = mean_squared_error(actual, predictions)
        print('mse = %.3f' % mse)
        save_file(save_path + 'mse.xls', row, col, '%.3f' % mse)

        rmse = sqrt(mse / len(predictions))
        print('rmse = %.3f' % rmse)
        save_file(save_path + 'rmse.xls', row, col, '%.3f' % rmse)

    # pcc = pearsonr(actual, predictions)[0]
    # print('pcc = %.3f' % pcc)
    # save_file(save_path + 'pcc.xls', row, col, '%.3f' % pcc)

    # ccc = 2 * pcc * sqrt(var(actual)) * sqrt(var(predictions)) / (
    # var(actual) + var(predictions) + (mean(actual) - mean(predictions)) ** 2)
    # print('ccc = %.3f' % ccc)
    # save_file(save_path + 'ccc.xls', row, col, '%.3f' % ccc)