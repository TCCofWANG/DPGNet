import argparse
import os
from time import strftime
from typing import Callable
from time import strftime
import os
from .dist import is_master_process
import shutil

#TODO Function to create the output directory
def create_output_dir(args):
    if args.output_dir==None:
        # Store the corresponding hyperparameters
        if args.save_log and is_master_process(): 

            args.output_dir = os.path.join('experiments')
            # Find the largest exp index and increment it by one
            current_exp = 0
            if os.path.exists(args.output_dir):
                exp_values = [int(f[3:]) for f in os.listdir(args.output_dir) if f.startswith('exp')]
                current_exp = max(exp_values) + 1 if exp_values else 0

            if args.exp_num != -1 and args.exp_num < current_exp:
                current_exp = args.exp_num

            args.output_dir = os.path.join(args.output_dir, 'exp{}'.format(current_exp))
    else:
        if not os.path.exists(args.output_dir):
            print('The output path {0} does not exist'.format(args.output_dir))
        else:
            shutil.rmtree(args.output_dir) # Delete all files in the corresponding folder

    current_time = strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, '{}.time'.format(current_time)), 'a+') as f:
        pass

    with open(os.path.join(args.output_dir, 'README'), 'a+') as f:
        f.write(args.desc)

    return args

def base_config(parser):
    # Settings unrelated to hyperparameters
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--no_pin_memory', action='store_true')

    parser.add_argument('--dist_url', type=str, default='env://')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--exp_num', type=int, default=-1)

    parser.add_argument('--desc', type=str, default='',
                        help='describe current experiment and save it into your exp{num}')
    return parser

def task_config(parser):
    # Settings related to training
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--end_epoch', type=int, default=100)
    parser.add_argument('--clip_max_norm', type=int, default=-1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--use_16bit', action='store_true')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--dataset_dir', type=str, default='./datasets/')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    return parser


def get_args(add_config: Callable = None):
    parser = argparse.ArgumentParser()
    if add_config is not None:
        parser = add_config(parser) # Hyperparameters set in main
    parser = base_config(parser)
    parser = task_config(parser)

    args = parser.parse_args() # Encapsulate the parameters

    args.save_log = True if args.no_log is False else False
    args.pin_memory = True if args.no_pin_memory is False else False

    return args
