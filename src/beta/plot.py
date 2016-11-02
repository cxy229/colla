# -*- coding: utf-8 -*-

from collamodel import Colla
from collamodel import engine_colla as engine
from collamodel import session_colla as session
from sqlalchemy import Table, MetaData
from sqlalchemy.sql import select
import numpy as np
import pandas as pd

PLOT_DATA_DIR = './plot_data/'
PLOT_DIR = './plot/'


def heatmap(a, b):
    d_time = {}
    d_times= {}

    with engine.connect() as con:
        meta = MetaData(engine)
        colla = Table('Colla', meta, autoload=True)
        stm = select([colla]) # .where(colla.c.id <100)
        rs = con.execute(stm)
        print('start')
        for tmp in rs:
            if tmp[a] in d_time.keys():
                d_time[tmp[a]][0] += tmp[4]
                d_time[tmp[a]][1] += 1
            else:
                d_time[tmp[a]] = [tmp[4], 1]

            if tmp[a] in d_times.keys():
                d_times[tmp[a]][0] += tmp[5]
                d_times[tmp[a]][1] += 1
            else:
                d_times[tmp[a]] = [tmp[5], 1]

        return d_time, d_times


def write_file(time, times, var_name1, var_name2):
    file_name1 = var_name1 + ',' + var_name2 + ',' + '平均合作时长.txt'
    file_name2 = var_name1 + ',' + var_name2 + ',' + '平均合作时长, 数量.txt'
    with open(PLOT_DATA_DIR + file_name1, 'w') as f:
        for key in time.keys():
            if time[key][1] != 0:
                print('%d, %d, %.3f' % (key[0], key[1], time[key][0] / time[key][1]), file=f)
            else:
                print('error: n == 0')
    with open(PLOT_DATA_DIR + file_name2, 'w') as f:
        for key in time.keys():
            if time[key][1] != 0:
                print('%d, %d, %.3f, %d' % (key[0], key[1], time[key][0] / time[key][1], time[key][1]), file=f)
            else:
                print('error: n == 0')

    file_name1 = var_name1 + ',' + var_name2 + ',' + '平均合作次数.txt'
    file_name2 = var_name1 + ',' + var_name2 + ',' + '平均合作次数, 数量.txt'
    with open(PLOT_DATA_DIR + file_name1, 'w') as f:
        for key in times.keys():
            if times[key][1] != 0:
                print('%d, %d, %.3f' % (key[0], key[1], times[key][0] / times[key][1]), file=f)
            else:
                print('error: n == 0')
    with open(PLOT_DATA_DIR + file_name2, 'w') as f:
        for key in times.keys():
            if time[key][1] != 0:
                print('%d, %d, %.3f, %d' % (key[0], key[1], times[key][0] / times[key][1], times[key][1]), file=f)
            else:
                print('error: n == 0')

def write_file_1(time, times, var_name1):
    file_name1 = var_name1 + ',' + '平均合作时长.txt'
    file_name2 = var_name1 + ',' + '平均合作时长, 数量.txt'
    with open(PLOT_DATA_DIR + file_name1, 'w') as f:
        for key in time.keys():
            if key<2222:
                print('%d, %.3f' % (key, time[key][0] / time[key][1]), file=f)
    with open(PLOT_DATA_DIR + file_name2, 'w') as f:
        for key in time.keys():
            if key<2222:
                print('%d, %.3f, %d' % (key, time[key][0] / time[key][1], time[key][1]), file=f)

    file_name1 = var_name1 + ',' + '平均合作次数.txt'
    file_name2 = var_name1 + ',' + '平均合作次数, 数量.txt'
    with open(PLOT_DATA_DIR + file_name1, 'w') as f:
        for key in times.keys():
            if key<2222:
                print('%d, %.3f' % (key, times[key][0] / times[key][1]), file=f)
    with open(PLOT_DATA_DIR + file_name2, 'w') as f:
        for key in times.keys():
            if key<2222:
                print('%d, %.3f, %d' % (key, times[key][0] / times[key][1], times[key][1]), file=f)

import xlsxwriter
def write_file_1_excel(time, times, var_name1):
    file_name1 = var_name1 + ',' + '平均合作时长.txt'
    file_name2 = var_name1 + ',' + '平均合作时长, 数量.xlsx'
    workbook = xlsxwriter.Workbook(file_name2)
    worksheet = workbook.add_worksheet()
    data=[]
    with open(PLOT_DATA_DIR + file_name1, 'w') as f:
        for key in time.keys():
            if key<2222:
                print('%d, %.3f' % (key, time[key][0] / time[key][1]), file=f)
    with open(PLOT_DATA_DIR + file_name2, 'w') as f:
        for key in time.keys():
            if key<2222:
                data.append([key, time[key][0] / time[key][1], time[key][1]])
    row = 0
    col = 0

    # Iterate over the data and write it out row by row.
    for item1, item2, item3 in (data):
        print(item1,item2,item3)
        worksheet.write(row, col, item1)
        worksheet.write(row, col + 1, item2)
        worksheet.write(row, col + 2, item3)
        row += 1
    workbook.close()
    data = []
    file_name1 = var_name1 + ',' + '平均合作次数.txt'
    file_name2 = var_name1 + ',' + '平均合作次数, 数量.xlsx'
    workbook = xlsxwriter.Workbook(file_name2)
    worksheet = workbook.add_worksheet()
    with open(PLOT_DATA_DIR + file_name1, 'w') as f:
        for key in times.keys():
            if key<2222:
                print('%d, %.3f' % (key, times[key][0] / times[key][1]), file=f)
    with open(PLOT_DATA_DIR + file_name2, 'w') as f:
        for key in times.keys():
            if key<2222:
                data.append([key, times[key][0] / times[key][1], times[key][1]])
    row = 0
    col = 0

    # Iterate over the data and write it out row by row.
    for item1, item2, item3 in (data):
        worksheet.write(row, col, item1)
        worksheet.write(row, col + 1, item2)
        worksheet.write(row, col + 2, item3)
        row += 1
    workbook.close()

def draw_plot(columns_time, l_time, interval, max=22222):
    import seaborn as sns
    sns.set()
    l_time = [i for i in l_time if (i[0]<=max and i[1]<=max)]
    df_time = pd.DataFrame(l_time, columns=columns_time)
    df_time = df_time.pivot(columns_time[0], columns_time[1], columns_time[2])
    ax_time = sns.heatmap(df_time,cmap='gist_rainbow', robust='True', xticklabels=interval, yticklabels=interval, square=True)
    # ax_time.set(ylim=(1834,0))
    ax_time.invert_yaxis()
    fig = ax_time.get_figure()
    fig.savefig(PLOT_DIR + columns_time[0] + '-' + columns_time[2] + '.pdf')

    # ax = sns.heatmap(, cmap="YlGnBu")

def read_plot_data(file_name):
    l_time = []
    with open(PLOT_DATA_DIR + file_name) as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split(', ')
            l_time.append([ int(line[0]), int(line[1]), float(line[2]) ])
    return l_time

def line_data(a):
    d_time = {}
    d_times = {}
    with engine.connect() as con:
        meta = MetaData(engine)
        colla = Table('Colla', meta, autoload=True)
        stm = select([colla]) # .where(colla.c.id <10)
        rs = con.execute(stm)
        print('start')
        for tmp in rs:
            if tmp[a] in d_time.keys():
                d_time[tmp[a]][0] += tmp[4]
                d_time[tmp[a]][1] += 1
            else:
                d_time[tmp[a]] = [tmp[4], 1]

            if tmp[a] in d_times.keys():
                d_times[tmp[a]][0] += tmp[5]
                d_times[tmp[a]][1] += 1
            else:
                d_times[tmp[a]] = [tmp[5], 1]
    return d_time, d_times

def read_plot_data_1(file_name):
    d = {}
    with open(PLOT_DATA_DIR + file_name) as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split(', ')
            d[int(line[0])] = float(line[1])
    x = list(d.keys())
    x.sort()
    y = [d[key] for key in x]
    return x,y

def draw_line():
    pass


def read_times_peoplenum():
    time = np.array([0 for i in range(0,52)])
    times = np.array([0 for i in range(0,426)])
    with engine.connect() as con:
        meta = MetaData(engine)
        colla = Table('Colla', meta, autoload=True)
        stm = select([colla]) #.where(colla.c.id < 100)
        rs = con.execute(stm)
        for tmp in rs:
            time[tmp[4]] += 1
            times[tmp[5]] += 1
    with open('./cache/time.txt', 'w') as f:
        for i in range(0, len(time)):
            f.write(str(time[i])+'\n')
            # print(time[i])
    with open('./cache/times.txt', 'w') as f:
        for i in range(0, len(times)):
            f.write(str(times[i])+'\n')

def draw_times_peoplenum():
    y = []
    with open('./cache/times.txt', 'r') as f:
        t = f.readline()
        while t != '':
            y.append(int(t.strip()))
            t = f.readline()
    x = [i for i in range(0,len(y))]
    print(sum(y[2:]))
    # import pylab as pl
    # pl.plot(x[1:], y[1:], '-or')
    # pl.xlabel("collaboration times")
    # pl.ylabel('number of people')
    # pl.show()

# draw_times_peoplenum()

if __name__ == '__main__':
    # print("学术年龄")
    # time, times = heatmap(7, 8)
    # write_file(time, times, "学术年龄1", "学术年龄2")
    # print("文章数量")
    # time, times = heatmap(9, 10)
    # write_file(time, times, "文章数量1", "文章数量2")
    # print("度")
    # time, times = heatmap(15, 16)
    # write_file(time, times, "度1", "度2")

    # time, times = line_data(13)
    # write_file_1_excel(time, times, "共同邻居")
    #
    # time, times = line_data(14)
    # write_file_1_excel(time, times, "最短路径")

    # sns.set()
    # time, times = heatmap(7, 8)
    # l_time = []
    # l_times = []
    # columns_time = ['scientific_age1', 'scientific_age2', 'time']
    # columns_times = ['scientific_age1', 'scientific_age2', 'times']
    #
    # for key in time.keys():
    #     l_time.append([key[0], key[1], time[key][0] / time[key][1]])
    # for key in times.keys():
    #     l_times.append([key[0], key[1], times[key][0] / times[key][1]])
    #
    # df_time = pd.DataFrame(l_time, columns=columns_time)
    # df_time = df_time.pivot('scientific_age1', 'scientific_age2', 'time')
    # df_times = pd.DataFrame(l_times, columns=columns_times)
    # df_times = df_times.pivot('scientific_age1', 'scientific_age2', 'times')
    #
    # ax_time = sns.heatmap(df_time, cmap="YlGnBu", yticklabels=3)
    # # ax_times = sns.heatmap(df_times, cmap="YlGnBu")
    #
    # fig = ax_time.get_figure()
    # fig.savefig(PLOT_DIR + columns_time[0] + '-' + columns_time[2] + '.png')
    #
    # # fig = ax_times.get_figure()
    # # fig.savefig(PLOT_DIR + columns_times[0] + '-' + columns_times[2] )

    # l_time = read_plot_data('学术年龄1,学术年龄2,平均合作时长.txt')
    # columns_time = ['Academic age A', 'Academic age B', 'time']
    # draw_plot(columns_time, l_time, 5, 50)

    # l_time = read_plot_data('学术年龄1,学术年龄2,平均合作次数.txt')
    # columns_time = ['Academic age A', 'Academic age B', 'times']
    # draw_plot(columns_time, l_time, 5, 50)

    # l_time = read_plot_data('度1,度2,平均合作时长.txt')
    # columns_time = ['Degree A', 'Degree B', 'time']
    # draw_plot(columns_time, l_time, 30, 300)

    # l_time = read_plot_data('度1,度2,平均合作次数.txt')
    # columns_time = ['Degree A', 'Degree B', 'times']
    # draw_plot(columns_time, l_time, 30, 300)

    # l_time = read_plot_data('文章数量1,文章数量2,平均合作时长.txt')
    # columns_time = ['Number of publications A', 'Number of publications B', 'time']
    # draw_plot(columns_time, l_time, 30, 300)

    # l_time = read_plot_data('文章数量1,文章数量2,平均合作次数.txt')
    # columns_time = ['Number of publications A', 'Number of publications B', 'times']
    # draw_plot(columns_time, l_time, 30, 300)

    # x,y = read_plot_data_1('最短路径,平均合作时长.txt')
    # print(x)
    # import pylab as pl
    # pl.plot(x, y, '-or')
    # pl.xlabel("shortest path length")
    # pl.ylabel('collaboration time')
    # pl.show()

    # x,y = read_plot_data_1('最短路径,平均合作次数.txt')
    # print(x)
    # import pylab as pl
    # pl.plot(x, y, '-or')
    # pl.xlabel("shortest path length")
    # pl.ylabel('collaboration times')
    # pl.show()

    # x,y = read_plot_data_1('共同邻居,平均合作时长.txt')
    # print(x)
    # import pylab as pl
    # pl.plot(x, y, '-or')
    # pl.xlabel("common neighbors' number")
    # pl.ylabel('collaboration time')
    # pl.show()

    # x,y = read_plot_data_1('共同邻居,平均合作次数.txt')
    # print(x)
    # import pylab as pl
    # pl.plot(x, y, '-or')
    # pl.xlabel("common neighbors' number")
    # pl.ylabel('collaboration times')
    # pl.show()
    # pl.savefig(PLOT_DIR + 'out.pdf')
