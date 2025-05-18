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
# #########################################!!! Attention !!! #################################################
# Distributed training means running on different machines, non-distributed training means running on one machine. (Multiple cards on one machine does not count as distributed training)
# Note that when you create a new Tensor, if you create it at init time, you should use nn.Parameter() so that the Tensor will be put on the correct cuda.
# Even if the same model, but different dataset, the corresponding output path should be different, otherwise the new code will delete all files under the current path when running (within the train function).
# If there is no batch dimension before inputting the data into the model, then convert its type to numpy, otherwise the first dimension will be split in half (two cards) by default.
# In the build model when the input adj, at this time the adj is already normalized Laplace matrix
#—————————————————————————————————————————————————————————————————————————————————————————————————————————————

device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False

'''The other parameters are in the config file.'''
def add_config(parser):

    parser.add_argument('--exp_name', default='deep_learning', choices=['deep_learning'])
    parser.add_argument('--train', default=True,type=str2bool,choices=[True,False], help='train or not')
    parser.add_argument('--resume', default=False, type=str2bool, choices=[True, False], help='Load pre-trained model or not')
    parser.add_argument('--output_dir',type=str,default='exp20/',help='None will automatically +1 from the existing output file, if specified it will overwrite the existing output file.')
    parser.add_argument('--resume_dir', type=str,default='exp20/',help='Retrieve the location of the checkpoint')

    parser.add_argument('--dp_mode', type=str2bool, default=False,help='Does it run on multiple cards')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=32)
    # model settings
    parser.add_argument('--model_name', type=str, default='WAVGCRN',help=['gwnet','gwnet_official',
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

    # Settings related to model parameters
    parser.add_argument('--num_features',help="Input model's feature dimensions(will be automatically specified based on the dataset)")
    parser.add_argument('--time_features',help='Time step features of the input model (will be specified automatically based on the dataset)')
    parser.add_argument('--num_nodes', help='Input the number of model nodes (will be specified automatically based on the dataset)')
    parser.add_argument('--d_model',type=int,default=32,help='Hidden layer dimension 1')
    parser.add_argument('--d_ff',type=int,default=64, help='Hidden layer dimension 2')
    parser.add_argument('--num_gcn', type=int, default=10, help='Number of GCNs')
    parser.add_argument('--patch_len', type=int, default=6, help='length of patch_len')
    parser.add_argument('--stride', type=int, default=1, help='length of stride')
    parser.add_argument('--points_per_hour', type=int, default=12, help='How many data sampling points in an hour (related to the dataset)')

    # TimeMixer
    parser.add_argument('--down_sampling_window', type=int, default=2, help='Down-sample by a factor of 2; the output sequence length is half of the original')
    parser.add_argument('--down_sampling_layers', type=int, default=1, help='Number of down-sampling layers, including the original layer')

    # N_PatchTST
    parser.add_argument('--patch_node_num', type=int, default=65, help='Number of nodes per patch, depending on the dataset')
    parser.add_argument('--patch_node_stride', type=int, default=65, help='Stride between patches, depending on the dataset; patches do not overlap')
    parser.add_argument('--info', type=str, default='None', help='Experimental information')
    return parser

def preprocess_args(args):
    args.pin_memory = False # The way read data is accelerated in Dataloader
    return args

if __name__ == '__main__':
    args = tu.config.get_args(add_config) # Getting set hyperparameters
    args = preprocess_args(args)
    set_seed(args.seed) # Set random seed

    print(f"|{'=' * 101}|")
    # Use the __dict__ method to get a dictionary of arguments and traverse the dictionary afterwards
    for key, value in args.__dict__.items():
        # Because the arguments may not all be str, it is necessary to convert all the data to str
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
        raise print('No exp file with name {0}'.format(args.exp_name))
