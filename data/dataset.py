
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

def split_dataset(dataset: np.ndarray, split_rate=0.8):
    '''
    groups=1
    :param dataset: x: (L, N, C)
    :param split_rate: proportion of the dataset to use for training
    :return: train: (L, N, C), test: (L, N, C)
    '''
    total_seq_len, num_nodes, _ = dataset.shape
    train_size = int(total_seq_len * split_rate)
    train_dataset, test_dataset = dataset[ 0:train_size, ...],dataset[train_size:, ...]

    return train_dataset, test_dataset


class TSDataset(Dataset):
    def __init__(self, dataset,time_dataset, seq_len,pred_len, feature_range=(-1, 1),**kwargs):
        '''
        :param dataset: x:(total_L, N, C)
        :param seq_len: length of split sequence
        :param pred_len:length of pred sequence
        :param feature_range: range of min_max scalar
        '''
        self.feature_range = feature_range
        self.pred_len = pred_len
        self.seq_len=seq_len
        self.mean=kwargs.get("mean")
        self.std=kwargs.get("std")
        self.max_values=kwargs.get("max")
        self.min_values = kwargs.get("min")

        assert len(self.feature_range) == 2 and self.feature_range[1] > self.feature_range[0]

        self.total_seq_len = len(dataset)
        self.dataset=dataset
        _,self.num_features,self.num_nodes=dataset.shape
        self.time_dataset=time_dataset


    def __getitem__(self, item):
        '''
        :param item: index
        :return: x: (C,L,N) and label：(C,N,L)
        '''
        x_end=item+self.seq_len
        y_end=x_end+self.pred_len
        x=self.dataset[item:x_end]
        y=self.dataset[x_end:y_end]
        x_time=self.time_dataset[item:x_end]
        y_time = self.time_dataset[x_end:y_end]
        x,y=torch.FloatTensor(x),torch.FloatTensor(y)
        x_time_, y_time_ = torch.FloatTensor(x_time[:,1:,:].astype(np.float32)), torch.FloatTensor(y_time[:,1:,:].astype(np.float32))


        return x.permute(0,2,1),x_time_,y.permute(0,2,1),y_time_

    def __len__(self):
        return len(self.dataset) - self.seq_len - self.pred_len



class SubwayDataset(Dataset):
    def __init__(self, dataset,time_dataset, seq_len,pred_len, feature_range=(-1, 1),**kwargs):
        '''
        :param dataset: x:(total_L, N, C)
        :param seq_len: length of split sequence
        :param pred_len:length of pred sequence
        :param feature_range: range of min_max scalar
        '''
        self.feature_range = feature_range
        self.pred_len = pred_len
        self.seq_len=seq_len
        self.mean=kwargs.get("mean")
        self.std=kwargs.get("std")
        self.max_values=kwargs.get("max")
        self.min_values = kwargs.get("min")

        assert len(self.feature_range) == 2 and self.feature_range[1] > self.feature_range[0]

        self.total_seq_len = len(dataset)
        self.dataset=dataset
        _,self.num_features,self.num_nodes=dataset.shape
        self.time_dataset=time_dataset


    def __getitem__(self, item):
        '''
        :param item: index
        :return: x: (C,L,N) and label：(C,N,L)
        '''
        x_end=item+self.seq_len
        y_end=x_end+self.pred_len
        x=self.dataset[item:x_end]
        y=self.dataset[x_end:y_end]
        x_time=self.time_dataset[item:x_end]
        y_time = self.time_dataset[x_end:y_end]
        x,y=torch.FloatTensor(x),torch.FloatTensor(y)
        x_time_, y_time_ = torch.FloatTensor(x_time[:,1:,:].astype(np.float32)), torch.FloatTensor(y_time[:,1:,:].astype(np.float32))
        x_ori_time,y_ori_time = x_time[:,0,0].tolist(), y_time[:,0,0].tolist()

        return x,x_time_,y,y_time_

    def __len__(self):
        return len(self.dataset) - self.seq_len - self.pred_len
    
class SubwayDataset4LLM(Dataset):
    def __init__(self, dataset,time_dataset, seq_len,pred_len, feature_range=(-1, 1),**kwargs):
        '''
        :param dataset: x:(total_L, N, C)
        :param seq_len: length of split sequence
        :param pred_len:length of pred sequence
        :param feature_range: range of min_max scalar
        '''
        self.feature_range = feature_range
        self.pred_len = pred_len
        self.seq_len=seq_len
        self.mean=kwargs.get("mean")
        self.std=kwargs.get("std")
        self.max_values=kwargs.get("max")
        self.min_values = kwargs.get("min")

        assert len(self.feature_range) == 2 and self.feature_range[1] > self.feature_range[0]

        self.total_seq_len = len(dataset)
        self.dataset=dataset
        _,self.num_features,self.num_nodes=dataset.shape
        self.time_dataset=time_dataset


    def __getitem__(self, item):
        '''
        :param item: index
        :return: x: (C,L,N) and label：(C,N,L)
        '''
        x_end=item+self.seq_len
        y_end=x_end+self.pred_len
        x=self.dataset[item:x_end]
        y=self.dataset[x_end:y_end]
        x_time=self.time_dataset[item:x_end]
        y_time = self.time_dataset[x_end:y_end]
        x,y=torch.FloatTensor(x),torch.FloatTensor(y)
        x_time, y_time = torch.Tensor(x_time), torch.Tensor(y_time)

        return x,x_time,y,y_time

    def __len__(self):
        return len(self.dataset) - self.seq_len - self.pred_len




import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

def split_dataset(dataset: np.ndarray, split_rate=0.8):
    '''
    groups=1
    :param dataset: x: (L, N, C)
    :param split_rate: proportion of the dataset to use for training
    :return: train: (L, N, C), test: (L, N, C)
    '''
    total_seq_len, num_nodes, _ = dataset.shape
    train_size = int(total_seq_len * split_rate)
    train_dataset, test_dataset = dataset[ 0:train_size, ...],dataset[train_size:, ...]

    return train_dataset, test_dataset


class TSDataset(Dataset):
    def __init__(self, dataset,time_dataset, seq_len,pred_len, feature_range=(-1, 1),**kwargs):
        '''
        :param dataset: x:(total_L, N, C)
        :param seq_len: length of split sequence
        :param pred_len:length of pred sequence
        :param feature_range: range of min_max scalar
        '''
        self.feature_range = feature_range
        self.pred_len = pred_len
        self.seq_len=seq_len
        self.mean=kwargs.get("mean")
        self.std=kwargs.get("std")
        self.max_values=kwargs.get("max")
        self.min_values = kwargs.get("min")

        assert len(self.feature_range) == 2 and self.feature_range[1] > self.feature_range[0]

        self.total_seq_len = len(dataset)
        self.dataset=dataset
        _,self.num_features,self.num_nodes=dataset.shape
        self.time_dataset=time_dataset


    def __getitem__(self, item):
        '''
        :param item: index
        :return: x: (C,L,N) and label：(C,N,L)
        '''
        x_end=item+self.seq_len
        y_end=x_end+self.pred_len
        x=self.dataset[item:x_end]
        y=self.dataset[x_end:y_end]
        x_time=self.time_dataset[item:x_end]
        y_time = self.time_dataset[x_end:y_end]
        x,y=torch.FloatTensor(x),torch.FloatTensor(y)
        x_time_, y_time_ = torch.FloatTensor(x_time[:,1:,:].astype(np.float32)), torch.FloatTensor(y_time[:,1:,:].astype(np.float32))


        return x.permute(0,2,1),x_time_,y.permute(0,2,1),y_time_

    def __len__(self):
        return len(self.dataset) - self.seq_len - self.pred_len



class SubwayDataset(Dataset):
    def __init__(self, dataset,time_dataset, seq_len,pred_len, feature_range=(-1, 1),**kwargs):
        '''
        :param dataset: x:(total_L, N, C)
        :param seq_len: length of split sequence
        :param pred_len:length of pred sequence
        :param feature_range: range of min_max scalar
        '''
        self.feature_range = feature_range
        self.pred_len = pred_len
        self.seq_len=seq_len
        self.mean=kwargs.get("mean")
        self.std=kwargs.get("std")
        self.max_values=kwargs.get("max")
        self.min_values = kwargs.get("min")

        assert len(self.feature_range) == 2 and self.feature_range[1] > self.feature_range[0]

        self.total_seq_len = len(dataset)
        self.dataset=dataset
        _,self.num_features,self.num_nodes=dataset.shape
        self.time_dataset=time_dataset


    def __getitem__(self, item):
        '''
        :param item: index
        :return: x: (C,L,N) and label：(C,N,L)
        '''
        x_end=item+self.seq_len
        y_end=x_end+self.pred_len
        x=self.dataset[item:x_end]
        y=self.dataset[x_end:y_end]
        x_time=self.time_dataset[item:x_end]
        y_time = self.time_dataset[x_end:y_end]
        x,y=torch.FloatTensor(x),torch.FloatTensor(y)
        x_time_, y_time_ = torch.FloatTensor(x_time[:,1:,:].astype(np.float32)), torch.FloatTensor(y_time[:,1:,:].astype(np.float32))
        x_ori_time,y_ori_time = x_time[:,0,0].tolist(), y_time[:,0,0].tolist()

        return x,x_time_,y,y_time_

    def __len__(self):
        return len(self.dataset) - self.seq_len - self.pred_len
    
class SubwayDataset4LLM(Dataset):
    def __init__(self, dataset,time_dataset, seq_len,pred_len, feature_range=(-1, 1),**kwargs):
        '''
        :param dataset: x:(total_L, N, C)
        :param seq_len: length of split sequence
        :param pred_len:length of pred sequence
        :param feature_range: range of min_max scalar
        '''
        self.feature_range = feature_range
        self.pred_len = pred_len
        self.seq_len=seq_len
        self.mean=kwargs.get("mean")
        self.std=kwargs.get("std")
        self.max_values=kwargs.get("max")
        self.min_values = kwargs.get("min")

        assert len(self.feature_range) == 2 and self.feature_range[1] > self.feature_range[0]

        self.total_seq_len = len(dataset)
        self.dataset=dataset
        _,self.num_features,self.num_nodes=dataset.shape
        self.time_dataset=time_dataset


    def __getitem__(self, item):
        '''
        :param item: index
        :return: x: (C,L,N) and label：(C,N,L)
        '''
        x_end=item+self.seq_len
        y_end=x_end+self.pred_len
        x=self.dataset[item:x_end]
        y=self.dataset[x_end:y_end]
        x_time=self.time_dataset[item:x_end]
        y_time = self.time_dataset[x_end:y_end]
        x,y=torch.FloatTensor(x),torch.FloatTensor(y)
        x_time, y_time = torch.Tensor(x_time), torch.Tensor(y_time)

        return x,x_time,y,y_time

    def __len__(self):
        return len(self.dataset) - self.seq_len - self.pred_len




