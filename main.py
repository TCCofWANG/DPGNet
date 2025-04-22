# -*- coding: utf-8 -*-
import os
# os.chdir('')
import io
import sys
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import torch_utils as tu
from torch_utils.set_seed import set_seed
import torch
from exp_.exp import EXP

#————————————————————————————————————————————————————————————————————————————————————————————————————————————
# #########################################！！！注意事项！！！#################################################
# 分布式训练指的是在不同的机子上跑，非分布式训练指的是在一个机子上跑。（一个机子多卡不算分布式训练）
# 需要注意的点，新建一个Tensor的时候，如果在init时候创建的时候，要使用nn.Parameter()来，这样子Tensor才会被放到正确的cuda上
# 即使的同一个模型，但是是不同的数据集，对应的输出路径也要不同，否则新的代码跑(train函数内)的时候会把对当前路径下的所有的文件全部删除
# 数据输入模型前如果没有batch维度，那么就将其类型转为numpy，否则默认会将第一维度对半分（两个卡）
# 在build model的时候送入的adj，此时的adj就已经是归一化后的拉普拉斯矩阵
#—————————————————————————————————————————————————————————————————————————————————————————————————————————————

device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False

'''其他参数在config文件内'''
def add_config(parser):
    # exp_name选择：
    # deep_learning_interval-->'DGCN','ASTGCN','MSTGCN','HA'
    # deep_learning-->剩下的模型+

    parser.add_argument('--exp_name', default='deep_learning', choices=['deep_learning'], help='模型选用哪一个exp文件')
    parser.add_argument('--train', default=True,type=str2bool,choices=[True,False], help='是否进行训练')
    parser.add_argument('--resume', default=False, type=str2bool, choices=[True, False], help='是否读取预训练权重')
    parser.add_argument('--output_dir',type=str,default=None,help='为None会自根据现有的输出文件自动+1，如果指定就会覆盖现有的输出文件')
    parser.add_argument('--resume_dir', type=str,default=None,help='读取checkpoint的位置')

    parser.add_argument('--dp_mode', type=str2bool, default=False,help='是否在多卡上跑')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=32)
    # model settings
    parser.add_argument('--model', type=str, default='gpt2', help='Specify the PLM to use')
    parser.add_argument('--model_name', type=str, default='STIDGCN_dynamic_patch_graph',help=['gwnet','gwnet_official',
                                                                            'DPG_Net','DPG_Mixer''DPG_Mixer_V2','DPG_Mixer_V3','DPG_Mixer_gwt'
                                                                            'PGCN','PMC_GCN','STIDGCN','TESTAM','WAVGCRN','STD_PLM', 'STIDGCN_dynamic_patch_graph'])

    parser.add_argument('--gnn_type', type=str, default='dcn', choices=['dcn', 'gat', 'gcn'])
    parser.add_argument('--data_name', type=str, default='weather', choices=['PeMS-Bay','METR-LA','PEMS04','PEMS08',
                                                                             'electricity','weather'])
    parser.add_argument('--embedding_use', type=str, default=True, help='是否使用embedding')
    parser.add_argument('--norm_use', type=str, default=True, help='是否使用norm')

    # dataset settings
    parser.add_argument('--seq_len', type=int, default=168)
    parser.add_argument('--pred_len', type=int, default=3)

    # 与模型参数有关的设定
    parser.add_argument('--num_features',help='输入模型的特征维度(会自动根据数据集指定)')
    parser.add_argument('--time_features',help='输入模型的时间步特征（会自动根据数据集指定）')
    parser.add_argument('--num_nodes', help='输入模型节点个数(会自动根据数据集指定)')
    parser.add_argument('--d_model',type=int,default=32,help='隐藏层维度1')
    parser.add_argument('--d_ff',type=int,default=64, help='隐藏层维度2')
    parser.add_argument('--num_gcn', type=int, default=10, help='GCN的个数')
    parser.add_argument('--patch_len', type=int, default=6, help='patch_len的长度')
    parser.add_argument('--stride', type=int, default=1, help='stride的长度')

    # 这是exp_interval中的模型的输入数据，输入数据有不同的周期性，窗口大小一般等于预测长度
    parser.add_argument('--num_of_weeks', type=int, default=2,help='表示一次取多少个week周期窗口')
    parser.add_argument('--num_of_days', type=int, default=1,help='表示一次取多少个day周期窗口')
    parser.add_argument('--num_of_hours', type=int, default=2,help='表示一次取多少个hour周期窗口(近期)')
    parser.add_argument('--points_per_hour', type=int, default=12,help='一个小时内有多少个数据采样点(与数据集有关)')

    # TimeMixer
    parser.add_argument('--down_sampling_window', type=int, default=2, help='下采样一次，输出序列是原本序列的1/2长度')
    parser.add_argument('--down_sampling_layers', type=int, default=1, help='包括自身')

    # N_PatchTST
    parser.add_argument('--patch_node_num', type=int, default=65, help='取决于数据集')
    parser.add_argument('--patch_node_stride', type=int, default=65, help='取决于数据集，理论上来说窗口不重叠')
    parser.add_argument('--info', type=str, default='None', help='实验信息')
    return parser

def preprocess_args(args):
    args.pin_memory = False # Dataloader中读数据加速的方式
    return args

if __name__ == '__main__':
    args = tu.config.get_args(add_config) # 获得设定的超参数
    args = preprocess_args(args)
    set_seed(args.seed) # 设置随机数种子

    print(f"|{'=' * 101}|")
    # 使用__dict__方法获取参数字典，之后遍历字典
    for key, value in args.__dict__.items():
        # 因为参数不一定都是str，需要将所有的数据转成str
        print(f"|{str(key):>50s}|{str(value):<50s}|")
    print(f"|{'=' * 101}|")
    print(device)
    if args.exp_name=='deep_learning':
        exp=EXP(args)
        if args.train:
            exp.train()
        with torch.no_grad():
            exp.test()
    
    else:
        raise print('没有名字为{0}的exp文件'.format(args.exp_name))
