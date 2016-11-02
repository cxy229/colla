import pandas as pd



csv_path = '../../csv/colla_80_time.csv'
save_path = './result/80/data/'

data = pd.read_csv(csv_path)
# data = data[0:20]


def time_statics(data):
    time = data['colla_time'].values
    time_res = {}
    for i in time:
        if i in time_res.keys():
            time_res[i] += 1
        else:
            time_res[i] = 1

    with open(save_path + 'time_1,num.txt', 'w') as f:
        for i in sorted(time_res.keys()):
            print('%d: %d' %(i,time_res[i]))
            print('%d: %d' %(i,time_res[i]), file=f)


def times_statics(data):
    times = data['coarticle_num'].values
    times_res = {}
    for i in times:
        if i in times_res.keys():
            times_res[i] += 1
        else:
            times_res[i] = 1

    with open(save_path + 'times_1,num.txt', 'w') as f:
        for i in sorted(times_res.keys()):
            print('%d: %d' % (i, times_res[i]))
            print('%d: %d' % (i, times_res[i]), file=f)


if __name__ == '__main__':
    time_statics(data)
    times_statics(data)