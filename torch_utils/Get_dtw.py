import torch
from torch.utils.data import DataLoader
from data.dataset import split_dataset,SubwayDataset
from data.data_process import *
import numpy as np
from fastdtw import fastdtw
from tqdm import tqdm
import torch.nn as nn
from tslearn.clustering import TimeSeriesKMeans, KShape
from sklearn.preprocessing import StandardScaler
class get_dtw(nn.Module):
    def __init__(self, config,dtw=True,pattern_keys=True):
        super().__init__()
        self.config=config
        df,Time, _ = load_data(config)
        Time = get_time_features(Time)  # (total_len, N=1, C=5), same shape as the dataset
        Time = Time.reshape(-1, config.time_features, 1)  # (total_len,C=5,N=1)
        self.df = df
        self.time=Time
        self.output_dim=config.output_dim
        self.points_per_hour = config.points_per_hour
        self.time_intervals=3600//config.points_per_hour 
        if dtw==True:
            self.dtw_matrix = self._get_dtw() # Obtain the DTW distance matrix between nodes
        self.points_per_day = 24 * 3600 // self.time_intervals 
        self.cand_key_days=config.cand_key_days = 14
        self.s_attn_size =config.s_attn_size=  3
        self.n_cluster =config.n_cluster=  16
        self.cluster_max_iter=config.cluster_max_iter = 5
        self.cluster_method =config.cluster_method="kshape"
        self.dataset=config.data_name
        if pattern_keys==True:
            self.pattern_keys=self._get_pattern_key() # Obtain centroids


    '''Obtain the DTW distance matrix'''
    def _get_dtw(self): 
        cache_path = './datasets/cache/dtw_' + self.config.data_name + '.npy'
        if not os.path.exists(cache_path): 
            print(f'Since the file at path {cache_path} does not exist, calculating DTW distances between nodes')
            df=self.df
            data_mean = np.mean( 
                [df[24 * self.points_per_hour * i: 24 * self.points_per_hour * (i + 1)] 
                 for i in range(df.shape[0] // (24 * self.points_per_hour))], axis=0) 
            _,self.num_nodes,self.feature=df.shape
            dtw_distance = np.zeros((self.num_nodes, self.num_nodes))
            for i in tqdm(range(self.num_nodes)):
                for j in range(i, self.num_nodes):
                    dtw_distance[i][j], _ = fastdtw(data_mean[:, i, :], data_mean[:, j, :], radius=6) 
                    # Calculate DTW distances between stations for each day
            for i in range(self.num_nodes): # Construct a symmetric matrix
                for j in range(i):
                    dtw_distance[i][j] = dtw_distance[j][i]
            np.save(cache_path, dtw_distance)

        dtw_matrix = np.load(cache_path)
        print('Load DTW matrix from {}'.format(cache_path))
        return dtw_matrix

    def get_seq_traindata(self):
        train_dataset, _ = split_dataset(self.df, split_rate=0.8)  
        train_time_dataset, _ = split_dataset(self.time, split_rate=0.8)

        _, num_nodes, num_features = train_dataset.shape
        scaler = StandardScaler(with_mean=True, with_std=True)
        train_dataset = scaler.fit_transform(train_dataset.reshape(len(train_dataset), -1))  
        train_dataset = train_dataset.reshape(len(train_dataset), num_features, num_nodes, )
        train_dataset = SubwayDataset(train_dataset,train_time_dataset, self.config.seq_len, self.config.pred_len)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,drop_last=False)
        self.train_dataset=[]
        for batch_x,_,batch_y,_ in train_dataloader:
            self.train_dataset.append(batch_x)
        self.train_dataset=torch.concat(self.train_dataset,dim=0)
        return self.train_dataset

    def _get_pattern_key(self):
        self.pattern_key_file = os.path.join(  
            './datasets/cache/', 'pattern_keys_{}_{}_{}_{}_{}_{}'.format(
                self.cluster_method, self.dataset, self.cand_key_days, self.s_attn_size, self.n_cluster,
                self.cluster_max_iter))
        if not os.path.exists(self.pattern_key_file + '.npy'):
            print(f'Since the file at path {self.pattern_key_file} does not exist, calculating the centroids after clustering')

            self.train_dataset=self.get_seq_traindata() 
            cand_key_time_steps = self.cand_key_days * self.points_per_day 
            pattern_cand_keys = (self.train_dataset[:cand_key_time_steps, :self.s_attn_size, :self.output_dim, :].permute(0,3,1,2) # FIXME 为什么这里要在时间维度上取3
                                 .reshape(-1, self.s_attn_size, self.output_dim)) 
            print("Clustering...")
            if self.cluster_method == "kshape": 
                km = KShape(n_clusters=self.n_cluster, max_iter=self.cluster_max_iter).fit(pattern_cand_keys)
            else: 
                km = TimeSeriesKMeans(n_clusters=self.n_cluster, metric="softdtw", max_iter=self.cluster_max_iter).fit(
                    pattern_cand_keys)
            self.pattern_keys = km.cluster_centers_
            np.save(self.pattern_key_file, self.pattern_keys)
            print("Saved at file " + self.pattern_key_file + ".npy")
        else:
            self.pattern_keys = np.load(self.pattern_key_file + ".npy")  
            print("Loaded file " + self.pattern_key_file + ".npy")

        return self.pattern_keys



