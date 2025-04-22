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