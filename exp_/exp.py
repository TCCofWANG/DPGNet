from model import *
from torch_utils.load_wave_graph import *
from data.dataset import split_dataset
import torch_utils as tu
from torch_utils import Write_csv,earlystopping
from data.data_process import *
from data.get_data import build_dataloader, build_TS_dataloader
import torch
import torch.nn as nn
import numpy as np
import test
import yaml
from datetime import datetime



device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

class EXP():
    def __init__(self,args):
        assert args.resume_dir==args.output_dir
        self.agrs=args
        tu.dist.init_distributed_mode(args)  # 初始化分布式训练
        # 早停机制
        if args.output_dir==None or args.output_dir=='None' or args.output_dir=='none':
            args.output_dir = None
            tu.config.create_output_dir(args)  # 创建输出的目录
            args.resume_dir=args.output_dir
        else:
            args.output_dir = os.path.join('experiments',args.output_dir)
            args.resume_dir = os.path.join('experiments', args.resume_dir)
        output_path = os.path.join(args.output_dir,args.model_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.output_path = output_path
        resume_path = os.path.join(args.resume_dir,args.model_name)
        if not os.path.exists(resume_path):
            raise print('没有找到对应的读取预训练权重的路径')
        resume_path = os.path.join(resume_path, args.data_name + '_best_model.pkl')
        self.resume_path = resume_path

        # seed = tu.dist.get_rank() + args.seed  # 0+args.seed
        # tu.model_tool.seed_everything(seed)  # 设置随机数种子，便于可重复实验

        # train_sampler,test_sampler是分布式训练时使用的，未分布式训练时，其都为None
        if args.data_name in ['electricity','weather']:
            # get_data timeseries datasets
            (adj, self.train_dataloader, self.val_dataloader, self.test_dataloader,
             self.train_sampler, self.val_sampler, self.test_sampler) = build_TS_dataloader(args)

        else:
            # get_data SP datasets
            (adj, self.train_dataloader,self.val_dataloader, self.test_dataloader,
             self.train_sampler,self.val_sampler,self.test_sampler) = build_dataloader(args)



        self.adj=adj # TODO 这里的adj就是普通的adj

        # get_model
        self.build_model(args, adj)  # 得到对应的模型
        self.model.to(device)  # 送入对应的设备中

        self.model = tu.dist.ddp_model(self.model, [args.local_rank])  # 分布式训练模型，好像也没有发挥作用
        if args.dp_mode:
            self.model = nn.DataParallel(self.model)  # 分布训练，单机子多显卡
            print('using dp mode')

        # 模型训练所需的
        criterion = nn.MSELoss()
        self.criterion=criterion

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # 引入权重衰减的Adam
        self.optimizer=optimizer

        # 权重衰减：cos衰减
        lr_optimizer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.end_epoch,eta_min=args.lr / 1000)
        self.lr_optimizer=lr_optimizer

        path = os.path.join(output_path, args.data_name + '_best_model.pkl')
        self.early_stopping = earlystopping.EarlyStopping(path=path, optimizer=self.optimizer,
                                                          scheduler=self.lr_optimizer, patience=args.patience)

        if args.resume:
            print('加载预训练模型')
            try:
                dp_mode = args.args.dp_mode
            except AttributeError as e:
                dp_mode = True
            # FIXME 自动读取超参数 这里还不完善
            hparam_path = os.path.join(args.output_dir, 'hparam.yaml')
            with open(hparam_path, 'r') as f:
                hparam_dict = yaml.load(f, yaml.FullLoader)
                args.output_dir = hparam_dict['output_dir']

            # 读取最好的权重
            self.load_best_model(path=self.resume_path,args=args, distributed=dp_mode)

    '''建立模型'''
    def build_model(self,args,adj):

        if args.model_name == 'gwnet_official':
            args.layers=4
            self.model = GWNet_official(num_nodes=args.num_nodes,in_dim=args.num_features,supports=adj,
                                     out_dim=1,pred_len=args.pred_len, n_hidden=32, kernel_size=2, layers=args.layers, blocks=1,
                                     addaptadj=True)

        elif args.model_name=='PGCN':
            self.model = PGCN(supports=adj,in_dim=args.num_features,out_dim=args.pred_len)

        elif args.model_name=='PMC_GCN':
            time_num = 96  # SH
            num_layers = 3  # Number of ST Block
            heads = 4  # Number of Heads in MultiHeadAttention
            cheb_K = 2  # Order for Chebyshev Polynomials (Eq 2)
            dropout = 0
            forward_expansion = 4  # Dimension of Feed Forward Network: embed_size --> embed_size * forward_expansion --> embed_size
            self.model = PMC_GCN(args.seq_len,adj,args.num_features,args.d_model,time_num,num_layers,args.seq_len,args.pred_len,heads,cheb_K,forward_expansion, dropout)

        elif args.model_name == 'STIDGCN':
            args.n_hidden = 32
            args.granularity = 24*args.points_per_hour # 一天多少个采样点，5min采样一次的话那就是288
            self.model = STIDGCN(input_dim=args.num_features, num_nodes=args.num_nodes, channels=args.n_hidden, granularity=args.granularity,
                                 input_len=args.seq_len,output_len=args.pred_len,  points_per_hour=args.points_per_hour, dropout=0.1)
            
        elif args.model_name == 'STIDGCN_dynamic_patch_graph':
            args.n_hidden = 32
            args.granularity = 24 * args.points_per_hour  # 一天多少个采样点，5min采样一次的话那就是288
            self.model = STIDGCN_dynamic_patch_graph(args=args, supports=adj, input_dim=args.num_features,
                                                     num_nodes=args.num_nodes, channels=args.n_hidden,
                                                     granularity=args.granularity,
                                                     input_len=args.seq_len, output_len=args.pred_len,
                                                     points_per_hour=args.points_per_hour, dropout=0.1)

        # elif args.model_name == "STID":
        #     args.if_T_i_D=True # 是否使用日期编码
        #     args.if_D_i_W=True # 是否使用周编码
        #     args.if_node=True # 是否是节点
        #     args.day_of_week_size=7 # 选取一周的n天
        #     args.time_of_day_size=args.points_per_hour*24 #一天有几个时间步记录
        #     args.num_layer=3
        #     self.model = STID(args)


        elif args.model_name == 'TESTAM':
            self.model = TESTAM(args.num_features, args.num_nodes, args.seq_len, dropout=0.3, in_dim=args.num_features, out_dim=args.pred_len, hidden_size=32, layers=3, points_per_hour=args.points_per_hour)

        elif args.model_name == 'WAVGCRN':
            self.model = WavGCRN(batch_size=args.batch_size, gcn_depth=2,num_nodes=args.num_nodes,predefined_A=[adj],seq_length=args.seq_len,out_dim=args.num_features,
                 in_dim=args.num_features,output_dim=args.pred_len,list_weight=[0.05, 0.95, 0.475],cl_decay_steps=2000,
                 hidden_size=64,points_per_hour=args.points_per_hour)

        elif args.model_name == 'DPG_Net':
            args.layers = 4
            self.model = DPG_Net(num_nodes=args.num_nodes, in_dim=args.num_features, supports=adj,
                                                      out_dim=1, pred_len=args.pred_len, d_model=args.d_model,
                                                      d_ff=args.d_ff,
                                                      kernel_size=2, layers=args.layers, blocks=1, args=args)

        elif args.model_name == 'DPG_Mixer':
            args.layers = 4
            self.model = DPG_Mixer(num_nodes=args.num_nodes, in_dim=args.num_features, supports=adj,
                                 out_dim=1, pred_len=args.pred_len, d_model=args.d_model,
                                 d_ff=args.d_ff,
                                 kernel_size=2, layers=args.layers, blocks=1, args=args)

        elif args.model_name == 'DPG_Mixer_V2':
            args.layers = 4
            self.model = DPG_Mixer_V2(num_nodes=args.num_nodes, in_dim=args.num_features, supports=adj,
                                 out_dim=1, pred_len=args.pred_len, d_model=args.d_model,
                                 d_ff=args.d_ff,
                                 kernel_size=2, layers=args.layers, blocks=1, args=args)

        elif args.model_name == 'DPG_Mixer_V3':
            args.layers = 4
            self.model = DPG_Mixer_V3(num_nodes=args.num_nodes, in_dim=args.num_features, supports=adj,
                                 out_dim=1, pred_len=args.pred_len, d_model=args.d_model,
                                 d_ff=args.d_ff,
                                 kernel_size=2, layers=args.layers, blocks=1, args=args)

        elif args.model_name == 'DPG_Mixer_gwt':
            args.layers = 4
            if args.pred_len == 3:
                gwt_path = './PEMS08_3_gwt.pkl'
                ckpt = torch.load(gwt_path)
                nodevec_1 = ckpt['model']['nodevec1'].to('cpu')
                nodevec_2 = ckpt['model']['nodevec2'].to('cpu')
                adp = F.softmax(F.relu(torch.mm(nodevec_1,nodevec_2)), dim=1)

            if args.pred_len == 6:
                gwt_path = './PEMS08_6_gwt.pkl'
                ckpt = torch.load(gwt_path)
                nodevec_1 = ckpt['model']['nodevec1'].to('cpu')
                nodevec_2 = ckpt['model']['nodevec2'].to('cpu')
                adp = F.softmax(F.relu(torch.mm(nodevec_1,nodevec_2)), dim=1)

            if args.pred_len == 12:
                gwt_path = './PEMS08_12_gwt.pkl'
                ckpt = torch.load(gwt_path)
                nodevec_1 = ckpt['model']['nodevec1'].to('cpu')
                nodevec_2 = ckpt['model']['nodevec2'].to('cpu')
                adp = F.softmax(F.relu(torch.mm(nodevec_1, nodevec_2)), dim=1)

            self.model = DPG_Mixer_gwt(num_nodes=args.num_nodes, in_dim=args.num_features, supports=adj+adp,
                                 out_dim=1, pred_len=args.pred_len, d_model=args.d_model,
                                 d_ff=args.d_ff,
                                 kernel_size=2, layers=args.layers, blocks=1, args=args)
        
        else:
            raise NotImplementedError

    '''一个epoch下的代码'''
    def train_test_one_epoch(self,args,dataloader,adj,save_manager: tu.save.SaveManager,epoch,mode='train',max_iter=float('inf'),**kargs):
        if mode == 'train':
            self.model.train()
            self.optimizer.zero_grad()
        elif mode == 'test' or mode =='val':
            self.model.eval()
        else:
            raise NotImplementedError

        metric_logger = tu.metric.MetricMeterLogger() # 初始化一个字典，记录对应的训练的损失结果

        # Dataloader，只不过为了分布式训练因此多了部分的代码
        for index, unpacked in enumerate(
                metric_logger.log_every(dataloader, header=mode, desc=f'{mode} epoch {epoch}')):
            if index > max_iter:
                break
            seqs, seqs_time,targets,targets_time = unpacked # (B,L,C,N)
            seqs, targets = seqs.cuda().float(), targets.cuda().float()
            seqs_time, targets_time = seqs_time.cuda().float(), targets_time.cuda().float()
            seqs,targets=seqs.permute(0,2,3,1),targets.permute(0,2,3,1)# (B,L,C,N)
            seqs_time, targets_time = seqs_time.permute(0, 2, 3, 1), targets_time.permute(0, 2, 3, 1) #(B,C,N=1,L)
            # TODO 模型的输入和输出的维度都是(B,C,N,L).输出的特征维度默认为1
            self.adj = np.array(self.adj)# 如果不是array，那么送入model的时候第一个维度会被分成两半
            pred = self.model(seqs,self.adj,seqs_time=seqs_time,targets_time=targets_time,targets=targets,mode=mode,index=index,epoch=epoch)  # 输入模型
            if (args.model_name=='MegaCRN' or args.model_name=='TESTAM') and mode=='train':
                pred,loss_part=pred[0],pred[1]

            # 计算损失 TODO 默认计算的是第一个特征维度
            targets = targets[:, 0:1, ...]
            if pred.shape[1]!=1:
                pred = pred[:,0:1,...]

            loss = self.criterion(pred.to(targets.device), targets) # 0表示的是特征只取流量这一个特征(参考DGCN的源代码)

            # 检测nan
            if np.isnan(np.array(loss.detach().cpu())).any():
                print('Pred exist nan')
                break

            # 计算MSE、MAE损失
            mse = torch.mean(torch.sum((pred - targets) ** 2, dim=1).detach())
            mae = torch.mean(torch.sum(torch.abs(pred - targets), dim=1).detach())

            metric_logger.update(loss=loss, mse=mse, mae=mae)  # 更新训练记录

            step_logs = metric_logger.values()
            step_logs['epoch'] = epoch
            save_manager.save_step_log(mode, **step_logs)  # 保存每一个batch的训练loss

            if mode == 'train':
                if args.model_name=='MegaCRN':
                    loss = loss + loss_part
                loss.backward()
                # 梯度裁剪
                if args.clip_max_norm > 0:  # 裁剪值大于0
                    nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), args.clip_max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        epoch_logs = metric_logger.get_finish_epoch_logs()
        epoch_logs['epoch'] = epoch
        save_manager.save_epoch_log(mode, **epoch_logs)  # 保存每一个epoch的训练loss

        return epoch_logs

    def train(self):
        args = self.agrs
        if args.resume!=True:
            tu.config.create_output_dir(args)  # 创建输出的目录
            print('output dir: {}'.format(args.output_dir))
            start_epoch = 0
        else:
            start_epoch=self.start_epoch

        # 以下是保存超参数
        save_manager = tu.save.SaveManager(args.output_dir, args.model_name, 'mse', compare_type='lt', ckpt_save_freq=30)
        save_manager.save_hparam(args)

        max_iter = float('inf')  # 知道满足对应的条件才会停下来

        # 以下开始正式的训练
        for epoch in range(start_epoch, args.end_epoch):
            if tu.dist.is_dist_avail_and_initialized():  # 不进入下面的代码
                self.train_sampler.set_epoch(epoch)
                self.val_sampler.set_epoch(epoch)
                self.test_sampler.set_epoch(epoch)

            tu.dist.barrier()  # 分布式训练，好像也没作用

            # train
            self.train_test_one_epoch(args,self.train_dataloader,self.adj, save_manager, epoch, mode='train')

            self.lr_optimizer.step()  # lr衰减

            # val
            val_logs = self.train_test_one_epoch(args, self.val_dataloader, self.adj, save_manager, epoch, mode='val')

            # 检测nan
            if np.isnan(val_logs['mse']).any():
                print('Pred exist nan')
                break
            # test
            test_logs = self.train_test_one_epoch(args,self.test_dataloader,self.adj, save_manager, epoch,mode='test')


            # 早停机制
            self.early_stopping(val_logs['mse'], model=self.model, epoch=epoch)
            if self.early_stopping.early_stop:
                break
        # 训练完成 读取最好的权重
        try:
            dp_mode = args.args.dp_mode
        except AttributeError as e:
            dp_mode = True
        output_path = os.path.join(self.output_path, args.data_name + '_best_model.pkl')
        self.load_best_model(path=output_path, args=args, distributed=dp_mode)


    def ddp_module_replace(self,param_ckpt):
        return {k.replace('module.', ''): v.cpu() for k, v in param_ckpt.items()}

    # TODO 加载最好的模型
    def load_best_model(self, path, args=None, distributed=True):

        ckpt_path = path
        if not os.path.exists(ckpt_path):
            print('路径{0}不存在，模型的参数都是随机初始化的'.format(ckpt_path))
        else:
            ckpt = torch.load(ckpt_path)

            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.lr_optimizer.load_state_dict(ckpt['lr_scheduler'])
            self.start_epoch=ckpt['epoch']

    def test(self):
        args=self.agrs
        try:
            dp_mode = args.args.dp_mode
        except AttributeError as e:
            dp_mode = True

        # 读取最好的权重
        if args.resume:
            self.load_best_model(path=self.resume_path, args=args, distributed=dp_mode)
        star = datetime.now()
        metric_dict=test.test(args,self.model,test_dataloader=self.test_dataloader,adj=self.adj)
        end=datetime.now()
        test_cost_time=(end-star).total_seconds()
        print("test花费了：{0}秒".format(test_cost_time))
        mae=metric_dict['mae']
        mse=metric_dict['mse']
        rmse=metric_dict['rmse']
        mape=metric_dict['mape']


        # 创建csv文件记录训练结果
        if not os.path.isdir('./results/'):
            os.mkdir('./results/')

        log_path = './results/experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR',
                          'batch_size', 'seed', 'MAE', 'MSE', 'RMSE','MAPE','seq_len',
                           'pred_len', 'd_model', 'd_ff','test_cost_time',
                           # 'e_layers', 'd_layers',
                            'info','output_dir']]
            Write_csv.write_csv(log_path, table_head, 'w+')

        time = datetime.now().strftime('%Y%m%d-%H%M%S')  # 获取当前系统时间
        a_log = [{'dataset': args.data_name, 'model': args.model_name, 'time': time,
                  'LR': args.lr,
                  'batch_size': args.batch_size,
                  'seed': args.seed, 'MAE': mae, 'MSE': mse,'RMSE':rmse,"MAPE":mape,'seq_len': args.seq_len,
                  'pred_len': args.pred_len,'d_model': args.d_model, 'd_ff': args.d_ff,
                  'test_cost_time': test_cost_time,
                  # 'e_layers': args.e_layers, 'd_layers': args.d_layers,
                  'info': args.info,'output_dir':args.output_dir}]
        Write_csv.write_csv_dict(log_path, a_log, 'a+')





