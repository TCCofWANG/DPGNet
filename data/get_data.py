from data.data_process import load_data,get_time_features,load_TS_data,get_TS_time_features
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from torch.utils import data
from data.dataset import split_dataset,SubwayDataset,TSDataset,SubwayDataset4LLM

'''Split dataset and create dataloader for TS dataset'''
def build_TS_dataloader(args, test=False):
    '''
    Training : Validation : Testing = 6:2:2 or 7:1:2. Training and validation have the same distribution, but testing differs.
    '''
    dataset,Time,adj = load_TS_data(args)  # obtain all data records and corresponding adjacency matrix
    Time=get_TS_time_features(Time) # shape (total_len, N=1, C=5), matching dataset
    total_len, num_nodes, num_features = dataset.shape  # shape(total_len,num_nodes,num_features)
    args.num_nodes = num_nodes  # store number of nodes
    args.num_features = num_features  # store number of features per node per timestep
    args.time_features = Time.shape[-1]  # store time feature dimension
    print('Number of nodes:', args.num_nodes)
    print('Feature dimension:', args.num_features)
    print('Time feature dimension:', args.time_features)


    # simply split train and test sets along sequence length dimension
    train_dataset,test_dataset = split_dataset(dataset,split_rate=0.8)
    Time = Time.reshape(-1, args.time_features, 1)  # (total_len,C=5,N=1)
    train_time_dataset, test_time_dataset = split_dataset(Time, split_rate=0.8)
    # next, split out training and validation sets: Training : Validation : Testing = 7:1:2
    train_dataset,val_dataset=split_dataset(train_dataset,split_rate=0.875)
    train_time_dataset, val_time_dataset = split_dataset(train_time_dataset, split_rate=0.875)

    # TODO: Here, test assuming distributions across stations are the same (common practice)
    train_len=len(train_dataset)
    val_len=len(val_dataset)
    test_len=len(test_dataset)

   # FIXME: Temporal scaling processing requires separating spatial and temporal data
    scaler = StandardScaler(with_mean=True,with_std=True)
    # ---------------------------------------------
    #   Feature naming needs to satisfy the following conditions:
    #   Can be adjusted here for different datasets.
    #   Should be improved to a universal format later,
    #   e.g. by directly obtaining column names.

    target_feature = 0
    train_dataset_norm = train_dataset[:, :, target_feature].copy()
    val_dataset_norm = val_dataset[:, :, target_feature].copy()
    test_dataset_norm = test_dataset[:, :, target_feature].copy()

    scaler = StandardScaler(with_mean=True, with_std=True)
    train_dataset_n = scaler.fit_transform(train_dataset_norm) 
    val_dataset_n = scaler.transform(val_dataset_norm)
    test_dataset_n = scaler.transform(test_dataset_norm)

    train_dataset[:,:,target_feature] = train_dataset_n
    val_dataset[:,:,target_feature] = val_dataset_n
    test_dataset[:,:,target_feature] = test_dataset_n
    mean = scaler.mean_.reshape(1, 1, num_nodes)
    std = scaler.scale_.reshape(1, 1, num_nodes)

    train_dataset = TSDataset(train_dataset, train_time_dataset, args.seq_len, args.pred_len,std=std,mean=mean)
    val_dataset = TSDataset(val_dataset, val_time_dataset, args.seq_len, args.pred_len,std=std,mean=mean)
    test_dataset = TSDataset(test_dataset, test_time_dataset, args.seq_len, args.pred_len,std=std,mean=mean)

    if not isinstance(adj, torch.Tensor):
        adj = torch.tensor(adj, dtype=torch.float32)
    adj.cuda()

    train_sampler, val_sampler,test_sampler = None,None,None


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                       drop_last=True,
                                       num_workers=args.num_workers,
                                       pin_memory=args.pin_memory)  

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                      drop_last=True,
                                      num_workers=args.num_workers, pin_memory=args.pin_memory)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                      drop_last=True,
                                      num_workers=args.num_workers, pin_memory=args.pin_memory)

   

    mean,std=np.expand_dims(mean,axis=-1),np.expand_dims(std,axis=-1) #(B=1,C,N,L=1)
    train_dataloader.mean,val_dataloader.mean,test_dataloader.mean=mean,mean,mean
    train_dataloader.std, val_dataloader.std, test_dataloader.std = std, std, std

    return adj, train_dataloader, val_dataloader, test_dataloader,None,None,None




'''Slicing the dataset and creating the dataloader'''
def build_dataloader(args, test=False):
    '''
    Training : Validation : Testing = 6:2:2 or 7:1:2. Training and validation have the same distribution, but testing differs.
    '''
    dataset,Time,adj = load_data(args)  # obtain all data records and corresponding adjacency
    Time=get_time_features(Time) #(total_len,N=1,C=5), matching dataset
    
    total_len, num_nodes, num_features = dataset.shape  # shape(total_len,num_nodes,num_features)
    args.num_nodes = num_nodes  
    args.num_features = num_features  
    args.time_features = Time.shape[-1]  
    print('Number of nodes:', args.num_nodes)
    print('Feature dimension:', args.num_features)
    print('Time feature dimension:', args.time_features)

    if args.data_name == "PeMS-Bay" or args.data_name == "METR-LA":
        train_dataset,test_dataset = split_dataset(dataset,split_rate=0.8)
        Time = Time.reshape(-1, args.time_features, 1)  # (total_len,C=5,N=1)
        train_time_dataset, test_time_dataset = split_dataset(Time, split_rate=0.8)
        train_dataset,val_dataset=split_dataset(train_dataset,split_rate=0.875)
        train_time_dataset, val_time_dataset = split_dataset(train_time_dataset, split_rate=0.875)
    else:
        train_dataset,test_dataset = split_dataset(dataset,split_rate=0.8)
        Time = Time.reshape(-1, args.time_features, 1)  # (total_len,C=5,N=1)
        train_time_dataset, test_time_dataset = split_dataset(Time, split_rate=0.8)
        train_dataset,val_dataset=split_dataset(train_dataset,split_rate=0.75)
        train_time_dataset, val_time_dataset = split_dataset(train_time_dataset, split_rate=0.75)

    # TODO: Here, test assuming distributions across stations are the same (common practice)
    train_len=len(train_dataset)
    val_len=len(val_dataset)
    test_len=len(test_dataset)
    if args.data_name == "PeMS-Bay" or args.data_name == "METR-LA":
        scaler = StandardScaler(with_mean=True, with_std=True)
        train_dataset = scaler.fit_transform(train_dataset.reshape(-1,num_features)) 
        val_dataset = scaler.transform(val_dataset.reshape(-1,num_features))
        test_dataset = scaler.transform(test_dataset.reshape(-1,num_features))

        train_dataset = train_dataset.reshape(train_len,num_features,num_nodes)
        val_dataset = val_dataset.reshape(val_len, num_features,num_nodes)
        test_dataset = test_dataset.reshape(test_len, num_features,num_nodes)
        mean = scaler.mean_.reshape(1, num_features,1)
        std = scaler.scale_.reshape(1, num_features,1)

        train_dataset = SubwayDataset(train_dataset, train_time_dataset, args.seq_len, args.pred_len,std=std,mean=mean)
        val_dataset = SubwayDataset(val_dataset, val_time_dataset, args.seq_len, args.pred_len,std=std,mean=mean)
        test_dataset = SubwayDataset(test_dataset, test_time_dataset, args.seq_len, args.pred_len,std=std,mean=mean)

    elif args.data_name=="PEMS04"or args.data_name=="PEMS08":
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_dataset = scaler.fit_transform(train_dataset.reshape(-1,num_features)) 
        val_dataset = scaler.transform(val_dataset.reshape(-1,num_features))
        test_dataset = scaler.transform(test_dataset.reshape(-1,num_features))

        train_dataset = train_dataset.reshape(train_len, num_features, num_nodes)
        val_dataset = val_dataset.reshape(val_len, num_features, num_nodes)
        test_dataset = test_dataset.reshape(test_len, num_features, num_nodes)

        min_values = scaler.data_min_.reshape(1, num_features,1)
        max_values = scaler.data_max_.reshape(1, num_features,1)
        train_dataset = SubwayDataset(train_dataset, train_time_dataset, args.seq_len, args.pred_len, max=max_values,min=min_values)
        val_dataset = SubwayDataset(val_dataset, val_time_dataset, args.seq_len, args.pred_len, max=max_values,min=min_values)
        test_dataset = SubwayDataset(test_dataset, test_time_dataset, args.seq_len, args.pred_len, max=max_values,min=min_values)
    else:
        assert print('Dataset normalization undefined')


    if not isinstance(adj, torch.Tensor):
        adj = torch.tensor(adj, dtype=torch.float32)
    adj.cuda()

    train_sampler, val_sampler,test_sampler = None,None,None

    if args.distributed and not test:
        train_sampler = data.DistributedSampler(train_dataset, seed=args.seed)  
        train_batch_sampler = data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                                           num_workers=args.num_workers, pin_memory=args.pin_memory)

        val_sampler = data.DistributedSampler(val_dataset, seed=args.seed)
        val_dataloader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler,
                                          num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False)

        test_sampler = data.DistributedSampler(test_dataset, seed=args.seed)
        test_dataloader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler,
                                          num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                           drop_last=True,
                                           num_workers=args.num_workers,
                                           pin_memory=args.pin_memory)  

        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                          drop_last=True,
                                          num_workers=args.num_workers, pin_memory=args.pin_memory)

        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                          drop_last=True,
                                          num_workers=args.num_workers, pin_memory=args.pin_memory)


    if args.data_name=="PeMS-Bay"or args.data_name=="METR-LA":
        mean,std=np.expand_dims(mean,axis=-1),np.expand_dims(std,axis=-1) #(B=1,C,N,L=1)
        train_dataloader.mean,val_dataloader.mean,test_dataloader.mean=mean,mean,mean
        train_dataloader.std, val_dataloader.std, test_dataloader.std = std, std, std

    elif args.data_name == "PEMS04" or args.data_name == "PEMS08" or args.data_name=="SZ_metro":
        min, max = np.expand_dims(min_values, axis=-1), np.expand_dims(max_values, axis=-1)  # (B=1,C,N,L=1)
        train_dataloader.min, val_dataloader.min, test_dataloader.min = min, min, min
        train_dataloader.max, val_dataloader.max, test_dataloader.max = max, max, max
    return adj, train_dataloader, val_dataloader, test_dataloader, train_sampler, val_sampler, test_sampler

