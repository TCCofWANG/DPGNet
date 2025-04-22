import pandas as pd
import numpy as np
import os
import csv
def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''

    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        # take the first two columns (from and to) to represent adjacency information
        edges = [(int(i[0]), int(i[1])) for i in reader] 

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    return A

def load_adjacency_matrix(distance_df_filename,num_of_vertices):
    adj = pd.read_pickle(distance_df_filename)[2]
    assert num_of_vertices==adj.shape[0]==adj.shape[1] and len(adj.shape)==2
    # The following processes the adjacency matrix: the loaded matrix has zeros on the diagonal and is weighted
    I=np.eye(num_of_vertices) # identity matrix
    adj=np.where(adj!=0,1,adj) # set any non-zero entries to 1, leave zeros unchanged
    adj=adj-I # set diagonal entries to 0
    return adj

def process_time_seq_data(dataset_dir,dataset_dir_adj):
    tmp=np.load(dataset_dir,allow_pickle=True)
    data=tmp['data'] # TODO: Feature data format is (total_len, N, C)
    time=tmp['time'] # TODO: Time part data format is (total_len,)
    assert data.shape[0]==time.shape[0]

    # TODO: The default feature dimensions are: first dimension is time, second is number of nodes, third is number of features
    total_len,node_num,dim=data.shape
    time=time.reshape(-1,1)
    # TODO: Here, the obtained adjacency matrix is an unweighted adjacency matrix
    if dataset_dir_adj.split('.')[-1]=='csv':
        adj=get_adjacency_matrix(dataset_dir_adj,num_of_vertices=node_num)
    elif dataset_dir_adj.split('.')[-1]=='pkl':
        adj = load_adjacency_matrix(dataset_dir_adj, num_of_vertices=node_num)
    else:
        raise print('Invalid adjacency matrix path. No file named {0}'.format(dataset_dir_adj))

    assert adj.shape[0]==adj.shape[1]==data.shape[1]

    return data,time,adj



def add_temporal_features(data, df):
    '''Add time of day and day of week as features to the data.'''
    l, n = data.shape
    feature_list = [np.expand_dims(data,axis=-1)]
    add_time_of_day = True  # Add time of day as a feature
    add_day_of_week = True  # Add day of the week as a feature
    add_day_of_month = True  # Add day of the month as a feature
    add_day_of_year = True  # Add day of the year as a feature
    steps_per_day = 24  # Number of time steps per day
    frequency = 1440 // steps_per_day
    df_index = pd.to_datetime(df['date'].values, format='%Y-%m-%d %H:%M:%S').to_numpy()
    df.index = df_index

    if add_time_of_day:
        # numerical time_of_day
        tod = [i % steps_per_day / steps_per_day for i in range(l)]
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        # numerical day_of_week
        dow = df.index.dayofweek / 7
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    if add_day_of_month:
        # numerical day_of_month
        dom = (df.index.day - 1) / 31 # df.index.day starts from 1. We need to minus 1 to make it start from 0.
        dom_tiled = np.tile(dom, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dom_tiled)

    if add_day_of_year:
        # numerical day_of_year
        doy = (df.index.dayofyear - 1) / 366 # df.index.month starts from 1. We need to minus 1 to make it start from 0.
        doy_tiled = np.tile(doy, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(doy_tiled)

    data_with_features = np.concatenate(feature_list, axis=-1)  # L x N x C
    return data_with_features


def process_time_seq_data_TS(dataset_dir):

    df = pd.read_csv(dataset_dir)
    df['date'] = pd.to_datetime(df['date'])
    time = df['date']
    fields = df.columns.values
    data = df[fields[1:]].values
    data = add_temporal_features(data,df)


    assert data.shape[0]==time.shape[0]

    # TODO: The default feature dimensions are: first dimension is time, second is number of nodes, third is number of features
    total_len,node_num,feature_num=data.shape
    dim = 1
    # For multivariate time series without a graph, initialize an identity matrix as adj; it does not affect subsequent GNN usage
    adj = np.eye(node_num)

    assert adj.shape[0]==adj.shape[1]==data.shape[1]

    return data,time,adj


def get_data_(dataset_dir,dataset_dir_adj):
    data,time, adj = process_time_seq_data(dataset_dir,dataset_dir_adj)

    return data,time,adj

def get_TS_data_(dataset_dir):
    data,time, adj = process_time_seq_data_TS(dataset_dir)

    return data,time,adj


'''This code reads data; by default, input feature data is stored in an npz file'''
def load_data(args):
    tmp=os.path.join(args.dataset_dir,args.data_name)
    dataset_dir = os.path.join(tmp,args.data_name+'.npz')

    # Determine the type of adjacency matrix file
    if 'distance.csv'in os.listdir(tmp):
        dataset_dir_adj = os.path.join(tmp,'distance.csv')
    elif 'adj.pkl'in os.listdir(tmp):
        dataset_dir_adj = os.path.join(tmp, 'adj.pkl')
    else:
        raise print('没有对应的adj文件')
    dataset,time, adj = get_data_(dataset_dir,dataset_dir_adj)
    assert len(dataset.shape)==3 and adj.shape[0]==adj.shape[1]

    return dataset,time, adj

'''This code reads data, modified for time series (TS) data'''
def load_TS_data(args):
    tmp=os.path.join(args.dataset_dir,args.data_name)
    dataset_dir = os.path.join(tmp,args.data_name+'.csv')

    dataset,time, adj = get_TS_data_(dataset_dir)
    assert len(dataset.shape)==3 and adj.shape[0]==adj.shape[1]

    return dataset,time, adj


'''Extract time features and normalize them'''
def get_TS_time_features(time):
    dt = pd.DataFrame({'dates': time})
    # Convert the date column to pandas Timestamp objects
    dt_ori = dt['dates'].to_numpy().reshape(-1, 1)
    dayofyear = dt['dates'].dt.dayofyear.values.reshape(-1, 1)  # The n day of the year
    dayofyear = (dayofyear - 1) / (365 - 1) - 0.5  # normalize to [-0.5, +0.5]
    dayofmonth = dt['dates'].dt.day.values.reshape(-1, 1)  # The n day of the mouth.
    dayofmonth = (dayofmonth - 1) / (31 - 1) - 0.5  # normalize to [-0.5, +0.5]
    dayofweek = dt['dates'].dt.dayofweek.values.reshape(-1, 1)  # The n day of the week
    dayofweek = (dayofweek - 0) / (6 - 0) - 0.5   # normalize to [-0.5, +0.5]
    hourofday = dt['dates'].dt.hour.values.reshape(-1, 1)  # The n hour of the day.
    hourofday = (hourofday - 0) / (23 - 0) - 0.5   # normalize to [-0.5, +0.5]
    minofhour = dt['dates'].dt.minute.values.reshape(-1, 1)  
    minofhour = (minofhour - 0) / (59 - 0) - 0.5   # normalize to [-0.5, +0.5]
    Time = np.concatenate((dt_ori.astype(str), dayofyear, dayofmonth, dayofweek, hourofday, minofhour), axis=-1)  # TODO 时间特征维度5维度
    time_feature = Time.shape[-1]
    Time = Time.reshape((-1, 1, time_feature))  # shape: (total_len, N=1, C=5) same shape as dataset
    return Time

'''Extract time features and normalize them'''
def get_time_features(time):
    dt = pd.DataFrame({'dates': time.flatten()})
    # Convert the date column to pandas Timestamp objects
    dt['dates'] = pd.to_datetime(dt['dates'])
    dt_ori = dt['dates'].to_numpy().reshape(-1, 1)
    dayofyear = dt['dates'].dt.dayofyear.values.reshape(-1, 1)  # The n day of the year
    dayofyear = (dayofyear - 1) / (365 - 1) - 0.5  # normalize to [-0.5, +0.5]
    dayofmonth = dt['dates'].dt.day.values.reshape(-1, 1) # The n day of the mouth.
    dayofmonth = (dayofmonth - 1) / (31 - 1) - 0.5  # normalize to [-0.5, +0.5]
    dayofweek = dt['dates'].dt.dayofweek.values.reshape(-1, 1)  # The n day of the week
    dayofweek = (dayofweek - 0) / (6 - 0) - 0.5  # normalize to [-0.5, +0.5]
    hourofday = dt['dates'].dt.hour.values.reshape(-1, 1)   # The n hour of the day.
    hourofday = (hourofday - 0) / (23 - 0) - 0.5  # normalize to [-0.5, +0.5]
    minofhour = dt['dates'].dt.minute.values.reshape(-1, 1)  # The n minute of the hour.
    minofhour = (minofhour - 0) / (59 - 0) - 0.5  # normalize to [-0.5, +0.5]
    Time = np.concatenate((dt_ori.astype(str), dayofyear, dayofmonth, dayofweek, hourofday, minofhour), axis=-1)  # TODO 时间特征维度5维度
    time_feature = Time.shape[-1]
    Time = Time.reshape((-1, 1, time_feature))  # shape: (total_len, N=1, C=5) same shape as dataset
    return Time
# 以下代码是为了部分模型的输入是多个不同周期所写的
'''找对应的数据'''
def search_data(sequence_length, num_of_batches, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data

    num_of_batches: int, the number of batches will be used for training 相当于取多少个对应的窗口，eg.取两个对应的窗口（一个窗口的大小与pred_len相同）

    label_start_idx: int, the first index of predicting target

    num_for_predict: int,
                     the number of points will be predicted for each sample

    units: int, week: 7 * 24, day: 24, recent(hour): 1（精确到小时即可，因为后续有计算一个小时有多少个采样点）。week：7天每一天24个小时，day：每一天24个小时

    points_per_hour: int, number of points per hour, depends on data 一个小时有多少个采样点

    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length: # 判断是否到末尾了
        return None

    x_idx = []
    for i in range(1, num_of_batches + 1):
        start_idx = label_start_idx - points_per_hour * units * i # ！！！这里会乘上一个小时有多少个采样点
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_batches:
        return None

    return x_idx[::-1]#倒叙输出,符合时间的 顺序输出,这里不占用多少空间
def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray 完整的数据集
                   shape is (sequence_length, num_of_vertices, num_of_features)

    num_of_weeks, num_of_days, num_of_hours: int

    label_start_idx: int, the first index of predicting target

    num_for_predict: int,（=pred_len）
                     the number of points will be predicted for each sample

    points_per_hour: int, default 12, number of points per hour（5分钟一个记录点）

    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)

    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)

    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)

    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                               label_start_idx, num_for_predict,
                               units=7 * 24, points_per_hour=points_per_hour)
    if not week_indices:
        return None

    day_indices = search_data(data_sequence.shape[0], num_of_days,
                              label_start_idx, num_for_predict,
                              24, points_per_hour)
    if not day_indices:
        return None

    hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                               label_start_idx, num_for_predict,
                               1, points_per_hour)
    if not hour_indices:
        return None

    week_sample = np.concatenate([data_sequence[i: j]
                                  for i, j in week_indices], axis=0)
    day_sample = np.concatenate([data_sequence[i: j]
                                 for i, j in day_indices], axis=0)
    hour_sample = np.concatenate([data_sequence[i: j]
                                  for i, j in hour_indices], axis=0)
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray

    Returns
    ----------
    stats: dict, two keys: mean and std

    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original

    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std=np.where(std==0,1,std) # 如果哪里的std为0，那我们设置其std为1。
    def normalize(x):
        return (x - mean) / std

    train = normalize(train)
    val = normalize(val)
    test =normalize(test)

    return {'mean': mean, 'std': std}, train, val, test
def read_and_generate_dataset(dataset,
                              num_of_weeks, num_of_days,
                              num_of_hours, num_for_predict,
                              points_per_hour=12, merge=False):
    '''
    Parameters
    ----------
    dataset: 完整的数据集
    num_of_weeks, num_of_days, num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data
    merge: boolean, default False,
           whether to merge training set and validation set to train model
    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_batches * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)
    '''
    data_seq = dataset# 读出直接是(total_len,node_num,in_feature)

    all_samples = []
    for idx in range(data_seq.shape[0]): # idx表示的是你的输入窗口的第一个位置，根据这个位置找不同的周期的信息
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour) # 得到不同的时间周期
        if not sample: # 一定是时间周期都有对应的值，该返回值才不是None
            continue

        week_sample, day_sample, hour_sample, target = sample
        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose((0,3,2,1)),
            np.expand_dims(day_sample, axis=0).transpose((0,3,2,1)),
            np.expand_dims(hour_sample, axis=0).transpose((0,3,2,1)),
            np.expand_dims(target, axis=0).transpose((0,3,2,1))
        )) # 将一个sample的值在新建一个维度，并且在那个维度上进行concat
    # FIXME 是不是这里的问题
    split_line1 = int(len(all_samples) * 0.6) # 训练集
    split_line2 = int(len(all_samples) * 0.8) # 测试集

    if not merge: # 进去这部分代码
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line1])]
    else:
        print('Merge training set and validation set!')
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line2])]

    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]
    # 训练集：验证集：测试集=6:2:2
    train_week, train_day, train_hour, train_target = training_set
    val_week, val_day, val_hour, val_target = validation_set
    test_week, test_day, test_hour, test_target = testing_set

    print('training data: week: {}, day: {}, recent: {}, target: {}'.format(
        train_week.shape, train_day.shape,
        train_hour.shape, train_target.shape))
    print('validation data: week: {}, day: {}, recent: {}, target: {}'.format(
        val_week.shape, val_day.shape, val_hour.shape, val_target.shape))
    print('testing data: week: {}, day: {}, recent: {}, target: {}'.format(
        test_week.shape, test_day.shape, test_hour.shape, test_target.shape))
    # 以下是进行标准化
    (week_stats, train_week_norm,
     val_week_norm, test_week_norm) = normalization(train_week,
                                                    val_week,
                                                    test_week)

    (day_stats, train_day_norm,
     val_day_norm, test_day_norm) = normalization(train_day,
                                                  val_day,
                                                  test_day)

    (recent_stats, train_recent_norm,
     val_recent_norm, test_recent_norm) = normalization(train_hour,
                                                        val_hour,
                                                        test_hour)

    (target_stats, train_target_norm,
     val_target_norm, test_target_norm) = normalization(train_target,
                                                        val_target,
                                                        test_target)


    all_data = {
        'train': {
            'week': train_week_norm,
            'day': train_day_norm,
            'recent': train_recent_norm,
            'target': train_target_norm,
        },
        'val': {
            'week': val_week_norm,
            'day': val_day_norm,
            'recent': val_recent_norm,
            'target': val_target_norm
        },
        'test': {
            'week': test_week_norm,
            'day': test_day_norm,
            'recent': test_recent_norm,
            'target': test_target_norm
        },
        'stats': {
            'week': week_stats,
            'day': day_stats,
            'recent': recent_stats,
            'target':target_stats
        }
    }

    return all_data