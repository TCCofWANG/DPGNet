import numpy as np
import torch
from matplotlib import pyplot as plt
import yaml
import os
import argparse
import torch_utils as tu
from tqdm import tqdm
from sklearn.metrics import r2_score, explained_variance_score,mean_absolute_percentage_error
from itertools import islice
import matplotlib.pyplot as plt
import seaborn as sns

torch.multiprocessing.set_sharing_strategy('file_system')

def np_eps(data, eps=1e-4):
    data[np.where(np.abs(data) < eps)] = eps
    return data


def smape_m(sources, outputs):
    mae=np.abs(sources-outputs)
    sources_=np.abs(sources)
    outputs_=np.abs(outputs)
    return np.mean(2*mae/(np_eps(sources_)+np_eps(outputs_)))



def mse_m(labels,preds, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(preds, labels)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        return np.mean(mse)


def mae_m(labels,preds, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)

# TODO 这个得到的不是百分比，就是小数，要得到百分比要乘上100
# def mape_m1(labels,preds, null_val=np.nan):
#     mape=mean_absolute_percentage_error(preds.flatten(), labels.flatten())
#     return mape

def mape_m(labels,preds, null_val=np.nan):
    tmp1=np.mean(labels)*0.1
    mape=np.abs(preds.flatten()- labels.flatten())/((labels+tmp1).flatten())
    return np.mean(mape)

def r2_m(sources, outputs):
    mse=np.square(sources - outputs)
    y_var=np.var(sources)# libcity源代码这样计算的
    y_var=1e-4 if y_var==0 else y_var
    return 1-np.mean(mse/y_var)


def explained_variance_score_np(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    return explained_variance_score(labels, preds)

def calc_metrics(sources, outputs,mean=None,std=None,max=None,min=None,rescale=True):
    '''outputs、sources:(total_len,C=1,num_nodes,pred_len)'''
    assert sources.shape==outputs.shape
    sources_denorm = sources.copy()
    outputs_denorm = outputs.copy()

    if np.all(mean!=None) and np.all(std!=None):
        if np.isnan(mean).all()==False and np.isnan(std).all()==False:
            if outputs.shape[1]==1:
                mean=mean[:,0:1,:] # TODO 这里取的是一个维度的特征
                std = std[:, 0:1, :]
            sources=sources*std+mean
            outputs=outputs*std+mean
        else:
            assert print("mean或者std有nan值")
    elif np.all(max!=None) and np.all(min!=None):
        if np.isnan(max).all()==False and np.isnan(min).all()==False:
            if outputs.shape[1]==1:
                min=min[:,0:1,:] # TODO 这里取的是一个维度的特征
                max = max[:, 0:1, :]
            # 由于范围是(-1~1)
            sources=(sources+1)/2*(max-min)+min
            outputs=(outputs+1)/2*(max-min)+min
        else:
            assert print("max或者min中有nan值")

    if rescale:
        mse = mse_m(sources, outputs)
        mae = mae_m(sources, outputs)
        rmse = np.sqrt(mse)
        r2 = r2_m(sources, outputs)
    else:
        mse = mse_m(sources_denorm, outputs_denorm)
        mae = mae_m(sources_denorm, outputs_denorm)
        rmse = np.sqrt(mse)
        r2 = r2_m(sources_denorm, outputs_denorm)
    mape = mape_m(sources, outputs)
    smape = smape_m(sources, outputs)
    metric_dict = {
        'mse': mse.item(),
        'mae': mae.item(),
        'rmse': rmse.item(),
        'r2': r2.item(),
        'mape': mape.item(),
        'smape': smape.item(),
    }

    return metric_dict


@torch.no_grad()
def pred_st_graph_data(model, dataloader,adj):
    '''
    :param model: spatio-temporal graph model in cuda
    :param dataloader: torch.utils.data.dataloader which shuffle and drop_last are 'False'
    :return: sources: (total_L, C, N, pred_len), outputs: (total_L, C, N, pred_len)
    total_L表示的是实例的个数
    '''
    sources = []
    outputs = []

    index=0
    _iter = tqdm(dataloader)
    for seqs,seqs_time,targets,targets_time in _iter:
        seqs, targets = seqs.cuda().float(), targets.cuda().float()
        seqs_time, targets_time = seqs_time.cuda().float(), targets_time.cuda().float()
        seqs, targets = seqs.permute(0, 2, 3, 1), targets.permute(0, 2, 3, 1)
        seqs_time, targets_time = seqs_time.permute(0, 2, 3, 1), targets_time.permute(0, 2, 3, 1)
        timestamp=torch.concatenate((seqs_time,targets_time),dim=-1)
        # 模型输入输出都是(B,C,N,L)
        seqs = seqs.cuda()
        pred = model(seqs,adj,seqs_time=seqs_time,targets_time=targets_time,targets=targets,mode='test',index=index,timestamp=timestamp,prompt_prefix=None) 
        index+=1
        pred = pred[0]
        if pred.shape[1]!=1:
            pred=pred[:,0:1,...]
        # print(pred.shape)
        # TODO 默认计算指标的是第一维特征
        sources.extend(targets[:,0:1,:,:].detach().cpu().numpy())
        outputs.extend(pred.detach().cpu().numpy())

    sources = np.array(sources)
    outputs = np.array(outputs)


    return sources, outputs


@torch.no_grad()
def vision_pred_st_graph_data(model, dataloader,adj):
    '''
    :param model: spatio-temporal graph model in cuda
    :param dataloader: torch.utils.data.dataloader which shuffle and drop_last are 'False'
    :return: sources: (total_L, C, N, pred_len), outputs: (total_L, C, N, pred_len)
    total_L表示的是实例的个数
    '''
    sources = []
    outputs = []


    # 1,2 表示不同尺度
    season_batch_graphs_0 = []
    trend_batch_graphs_0 = []
    seasons_list_0 = []
    seasons_list_1 = []

    season_batch_graphs_1 = []
    trend_batch_graphs_1 = []
    targets_batch_time = []
    index=0
    _iter = tqdm(islice(dataloader, 240))
    for seqs,seqs_time,targets,targets_time,seqs_ori_time,targets_ori_time in _iter:
        seqs, targets = seqs.cuda().float(), targets.cuda().float()
        seqs_time, targets_time = seqs_time.cuda().float(), targets_time.cuda().float()
        seqs, targets = seqs.permute(0, 2, 3, 1), targets.permute(0, 2, 3, 1)
        seqs_time, targets_time = seqs_time.permute(0, 2, 3, 1), targets_time.permute(0, 2, 3, 1)
        # 模型输入输出都是(B,C,N,L)
        seqs = seqs.cuda()
        pred,season_graphs,trend_graphs,seasons = model(seqs,adj,seqs_time=seqs_time,targets_time=targets_time,targets=targets,mode='test',index=index)
        index+=1
        if pred.shape[1]!=1:
            pred=pred[:,0:1,...]
        # print(pred.shape)
        # TODO 默认计算指标的是第一维特征
        sources.extend(targets[:,0:1,:,:].detach().cpu().numpy())
        outputs.extend(pred.detach().cpu().numpy())
        season_batch_graphs_0.extend(season_graphs[0].detach().cpu().numpy())
        season_batch_graphs_1.extend(season_graphs[1].detach().cpu().numpy())
        seasons_list_0.extend(seasons[0].detach().cpu().numpy())
        seasons_list_1.extend(seasons[1].detach().cpu().numpy())

        trend_batch_graphs_0.extend(trend_graphs[0].detach().cpu().numpy())
        trend_batch_graphs_1.extend(trend_graphs[1].detach().cpu().numpy())

        targets_batch_time.extend(targets_ori_time[0])

    sources = np.array(sources)
    outputs = np.array(outputs)

    season_graphs_0 = np.array(season_batch_graphs_0)
    season_graphs_1 = np.array(season_batch_graphs_1)

    trend_graphs_0 = np.array(trend_batch_graphs_0)
    trend_graphs_1 = np.array(trend_batch_graphs_1)

    seasons_0 = np.array(seasons_list_0)
    seasons_1 = np.array(seasons_list_1)

    graph_time = np.array(targets_batch_time)

    return sources, outputs, season_graphs_0, season_graphs_1, trend_graphs_0, trend_graphs_1, graph_time,seasons_0 ,seasons_1



def test(args,model,test_dataloader,adj):
    # tu.model_tool.seed_everything(args.seed, benchmark=False)

    for f in os.listdir(args.resume_dir):
        if f.startswith('_best_'):
            print('best checkpoint:{}'.format(f))
    print('args : {}'.format(args))

    model.eval()
    test_out_dir=os.path.join(args.output_dir,'test')
    os.makedirs(test_out_dir, exist_ok=True)

    sources, outputs = pred_st_graph_data(model, test_dataloader,adj)
    torch.cuda.empty_cache() # 清空GPU的缓存
    # FIXME SZ_metro是最大最小归一化还是标准化
    if args.data_name=="PeMS-Bay"or args.data_name=="METR-LA":
        mean=test_dataloader.mean # 得到对应的标准化的均值和方差，进行反标准化
        std=test_dataloader.std
        metric_dict = calc_metrics(sources, outputs,mean=mean,std=std)
    elif args.data_name == "PEMS04" or args.data_name == "PEMS08" or args.data_name=="SZ_metro":
        min=test_dataloader.min
        max=test_dataloader.max
        metric_dict = calc_metrics(sources, outputs,min=min,max=max)
    elif args.data_name in ['electricity','weather']:
        mean=test_dataloader.mean # 得到对应的标准化的均值和方差，进行反标准化
        std=test_dataloader.std
        metric_dict = calc_metrics(sources, outputs,mean=mean,std=std,rescale=False)
    else:
        assert print("数据集规范化未定义")

    for k, v in metric_dict.items():
        metric_dict[k] = round(v, 5) # 四舍五入

    print(metric_dict) # 打印结果

    with open(os.path.join(test_out_dir, '{}_metric.yaml'.format(args.train)), 'w+') as f:
        f.write(yaml.dump(metric_dict))

    # visualize_plt(sources, outputs, test_out_dir, args.train_test, max_show=args.max_show)
    return metric_dict


def curve_vision(seasons, times, info, save_path):

    path = os.path.join(save_path,f'{info}_curve_season_collection')
    os.makedirs(os.path.join(save_path,f'{info}_curve_season_collection'), exist_ok=True)

    # 确保save_path存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 提取第一个特征维度的数据
    first_feature_data = seasons[0,:, :, 0]  # shape: (207, 12)

    # # 遍历每个站点，绘制时间曲线
    # for i in range(first_feature_data.shape[0]):  # 对每个站点遍历
    #     current_time = times[0].split(".")[0].replace(":", "_")
    #     # 获取当前站点的时间序列数据
    #     data = first_feature_data[i]
    #
    #     # 创建一个新的图形
    #     plt.figure(figsize=(10, 6))
    #
    #     # 绘制当前站点的时间曲线
    #     plt.plot(data, label=f"Station {i + 1}", color='b')
    #
    #     # 设置图形标题和标签
    #     plt.title(f"{current_time} Time Curve for Station {i + 1}")
    #     plt.xlabel("Time")
    #     plt.ylabel("First Feature Value")
    #
    #     # 显示网格
    #     plt.grid(True)
    #
    #     # 保存图像到指定路径
    #     plt.tight_layout()  # 自动调整子图参数，避免标签被遮挡
    #     plt.savefig(os.path.join(path, f"station_{i + 1}_curve.png"))
    #
    #     # 清空当前图形，以便绘制下一个站点的图像
    #     plt.close()
    #
    # print(f"All season-block outputs for station curves have been saved to {path}")

    return

def graphs_vision(sources,graphs,times,info,save_path):
    _,_,_,o = sources.shape
    B,N,N = graphs.shape
    # 设置图像大小
    plt.figure(figsize=(8, 8))

    # 创建保存图像的文件夹
    path = os.path.join(save_path,f'{info}_image_collection')
    os.makedirs(os.path.join(save_path,f'{info}_image_collection'), exist_ok=True)

    # 遍历 100 张图像
    # for i in range(50):
    #     # 获取当前图像和物理时间
    #     current_image = graphs[i, :, :]
    #     current_time = times[i].split(".")[0].replace(":", "_")
    #     current_info = info
    #
    #     # 绘制图像
    #     plt.figure(figsize=(8, 8))
    #     sns.heatmap(current_image, annot=False, cmap='YlOrRd')
    #     plt.axis('off')
    #     plt.title(f"Time: {current_time}\nName: {current_info}")
    #
    #     # 保存图像
    #     plt.savefig(os.path.join(path, f"{current_time}.png"))
    #     plt.close()
    #print(f"All season graph attentions for station curves have been saved to {path}")
    #

    path = os.path.join(save_path,f'{info}_curve_collection')
    os.makedirs(os.path.join(save_path,f'{info}_curve_collection'), exist_ok=True)


    i = 0
    for j in range(N):
        # 提取该位置所有矩阵的对应值
        values_at_position = graphs[:, i, j]

        # 规范值域范围到[-1,1]
        values_at_position = min_max_normalize(values_at_position)

        # 绘制曲线图
        plt.plot(values_at_position)
        plt.title(f"Values at position ({i}, {j}) over {B} graphs")
        plt.xlabel("time Index")
        plt.ylabel("Value at Position")
        # 保存图像
        plt.savefig(os.path.join(path, f"({i}, {j}).png"))
        plt.close()
    print(f"All season-correlations  between stations curve have been saved to {path}")


    # TODO 计算sources站点间的余弦相似性
    sources_cosine = calculate_cosine_similarity(sources)


    path = os.path.join(save_path,f'{info}_groundTruth_curve_collection')
    os.makedirs(os.path.join(save_path,f'{info}_groundTruth_curve_collection'), exist_ok=True)

    i = 0
    for j in range(N):
        # 提取该位置所有矩阵的对应值
        values_at_position = sources_cosine[:, i, j]

        # 绘制曲线图
        plt.plot(values_at_position)
        plt.title(f"GT Values at position ({i}, {j}) over {B} graphs")
        plt.xlabel("time Index")
        plt.ylabel("GT Value at Position")
        # 保存图像
        plt.savefig(os.path.join(path, f"({i}, {j}).png"))
        plt.close()
    print(f"All season-correlations GroundTruth between stations curve have been saved to {path}")


    return


def min_max_normalize(arr):
    # 计算原始数组的最小值和最大值
    min_val = np.min(arr)
    max_val = np.max(arr)

    # 归一化到 [0, 1] 范围
    normalized = (arr - min_val) / (max_val - min_val)

    # 将值域映射到 [-1, 1]
    normalized = 2 * normalized - 1

    return normalized

def calculate_cosine_similarity(sources):
    """
    计算每个站点之间的余弦相似性，并将结果存储在一个维度为 [240, 207, 207] 的矩阵中。

    Parameters:
    sources (ndarray): 形状为 [240, 1, 207, 12] 的 numpy 数组，表示每个站点的窗口数据。

    Returns:
    ndarray: 形状为 [240, 207, 207] 的余弦相似性矩阵。
    """
    # 对于每个批次
    batch_size, _, num_stations, window_length = sources.shape

    # 提取数据：形状 [240, 207, 12]
    data = sources[:, 0, :, :]  # 获取所有批次的数据

    # 计算每个站点的 L2 范数 (norm)
    norms = np.linalg.norm(data, axis=2, keepdims=True)  # 计算每个站点在窗口维度上的L2范数

    # 计算每对站点的余弦相似性
    dot_product = np.matmul(data, data.transpose(0, 2, 1))  # 点积：形状 [240, 207, 207]

    # 计算余弦相似性：点积 / (L2范数的乘积)
    cosine_similarities = dot_product / (norms * norms.transpose(0, 2, 1))  # 形状 [240, 207, 207]

    return cosine_similarities





def vision_test(args,model,test_dataloader,adj):
    # tu.model_tool.seed_everything(args.seed, benchmark=False)

    for f in os.listdir(args.resume_dir):
        if f.startswith('_best_'):
            print('best checkpoint:{}'.format(f))
    print('args : {}'.format(args))

    model.eval()
    test_out_dir=os.path.join(args.output_dir,'test')
    os.makedirs(test_out_dir, exist_ok=True)

    sources, outputs, season_graphs_0, season_graphs_1, trend_graphs_0, trend_graphs_1, graph_time, seasons_0, seasons_1 = vision_pred_st_graph_data(model, test_dataloader,adj)
    curve_vision(seasons_0,graph_time,'seasons_0',test_out_dir)
    graphs_vision(sources,season_graphs_0,graph_time,'seasons_0',test_out_dir)

    torch.cuda.empty_cache() # 清空GPU的缓存
    # FIXME SZ_metro是最大最小归一化还是标准化
    if args.data_name=="PeMS-Bay"or args.data_name=="METR-LA":
        mean=test_dataloader.mean # 得到对应的标准化的均值和方差，进行反标准化
        std=test_dataloader.std
        metric_dict = calc_metrics(sources, outputs,mean=mean,std=std)
    elif args.data_name == "PEMS04" or args.data_name == "PEMS08" or args.data_name=="SZ_metro":
        min=test_dataloader.min
        max=test_dataloader.max
        metric_dict = calc_metrics(sources, outputs,min=min,max=max)
    else:
        assert print("数据集规范化未定义")

    for k, v in metric_dict.items():
        metric_dict[k] = round(v, 5) # 四舍五入

    print(metric_dict) # 打印结果

    with open(os.path.join(test_out_dir, '{}_metric.yaml'.format(args.train)), 'w+') as f:
        f.write(yaml.dump(metric_dict))

    # visualize_plt(sources, outputs, test_out_dir, args.train_test, max_show=args.max_show)
    return metric_dict

@torch.no_grad()
def pred_st_graph_data_interval(model, dataloader,adj):
    '''
    :param model: spatio-temporal graph model in cuda
    :param dataloader: torch.utils.data.dataloader which shuffle and drop_last are 'False'
    :return: sources: (total_L, pred_L, N, C), outputs: (total_L, pred_L, N, C)
    total_L表示的是实例的个数
    '''
    sources = []
    outputs = []

    _iter = tqdm(dataloader)

    for train_w, train_d, train_r, targets in _iter:
        train_w = train_w.to('cuda').float()
        train_d = train_d.to('cuda').float()
        train_r = train_r.to('cuda').float()
        targets = targets.to('cuda').float()
        pred = model(train_w,train_d,train_r,adj)

        if pred.shape[1] != 1:
            pred = pred[:, 0:1, ...]
        # TODO 默认指标计算使用的是第一个特征
        sources.extend(targets[:,0:1,...].cpu().detach().numpy())
        outputs.extend(pred.detach().cpu().numpy())

    sources = np.array(sources)
    outputs = np.array(outputs)

    return sources, outputs


def test_interval(args,model,test_dataloader,adj):
    # tu.model_tool.seed_everything(args.seed, benchmark=False)

    for f in os.listdir(args.resume_dir):
        if f.startswith('_best_'):
            print('best checkpoint:{}'.format(f))
    print('args : {}'.format(args))

    test_out_dir=os.path.join(args.output_dir,'test')
    os.makedirs(test_out_dir, exist_ok=True)

    sources, outputs = pred_st_graph_data_interval(model, test_dataloader,adj)
    torch.cuda.empty_cache() # 清空GPU的缓存

    # total_seq_len = test_dataloader.total_seq_len - test_dataloader.seq_len - 1
    mean=test_dataloader.mean
    std=test_dataloader.std
    metric_dict = calc_metrics(sources, outputs,mean=mean,std=std)
    for k, v in metric_dict.items():
        metric_dict[k] = round(v, 5) # 四舍五入

    print(metric_dict) # 打印结果

    with open(os.path.join(test_out_dir, '{}_metric.yaml'.format(args.train)), 'w+') as f:
        f.write(yaml.dump(metric_dict))

    return metric_dict