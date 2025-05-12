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
        tu.dist.init_distributed_mode(args)  
        
        if args.output_dir==None or args.output_dir=='None' or args.output_dir=='none':
            args.output_dir = None
            tu.config.create_output_dir(args) 
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
            raise print('The corresponding path for reading the pre-trained weights was not found.')
        resume_path = os.path.join(resume_path, args.data_name + '_best_model.pkl')
        self.resume_path = resume_path


        if args.data_name in ['electricity','weather']:
            # get_data timeseries datasets
            (adj, self.train_dataloader, self.val_dataloader, self.test_dataloader,
             self.train_sampler, self.val_sampler, self.test_sampler) = build_TS_dataloader(args)

        else:
            # get_data SP datasets
            (adj, self.train_dataloader,self.val_dataloader, self.test_dataloader,
             self.train_sampler,self.val_sampler,self.test_sampler) = build_dataloader(args)



        self.adj=adj # TODO The adj here is just an ordinary adj.

        # get_model
        self.build_model(args, adj)  
        self.model.to(device)  

        self.model = tu.dist.ddp_model(self.model, [args.local_rank])  
        if args.dp_mode:
            self.model = nn.DataParallel(self.model)  
            print('using dp mode')


        criterion = nn.MSELoss()
        self.criterion=criterion

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  
        self.optimizer=optimizer

        lr_optimizer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.end_epoch,eta_min=args.lr / 1000)
        self.lr_optimizer=lr_optimizer

        path = os.path.join(output_path, args.data_name + '_best_model.pkl')
        self.early_stopping = earlystopping.EarlyStopping(path=path, optimizer=self.optimizer,
                                                          scheduler=self.lr_optimizer, patience=args.patience)

        if args.resume:
            print('Load the pre-trained model')
            try:
                dp_mode = args.args.dp_mode
            except AttributeError as e:
                dp_mode = True
            
            hparam_path = os.path.join(args.output_dir, 'hparam.yaml')
            with open(hparam_path, 'r') as f:
                hparam_dict = yaml.load(f, yaml.FullLoader)
                args.output_dir = hparam_dict['output_dir']

            # load the best model
            self.load_best_model(path=self.resume_path,args=args, distributed=dp_mode)

    '''Build the model'''
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
            args.granularity = 24*args.points_per_hour 
            self.model = STIDGCN(input_dim=args.num_features, num_nodes=args.num_nodes, channels=args.n_hidden, granularity=args.granularity,
                                 input_len=args.seq_len,output_len=args.pred_len,  points_per_hour=args.points_per_hour, dropout=0.1)
            



        elif args.model_name == 'TESTAM':
            self.model = TESTAM(args.num_features, args.num_nodes, args.seq_len, dropout=0.3, in_dim=args.num_features, out_dim=args.pred_len, hidden_size=32, layers=3, points_per_hour=args.points_per_hour)

        elif args.model_name == 'WAVGCRN':
            self.model = WavGCRN(batch_size=args.batch_size, gcn_depth=2,num_nodes=args.num_nodes,predefined_A=[adj],seq_length=args.seq_len,out_dim=args.num_features,
                 in_dim=args.num_features,output_dim=args.pred_len,list_weight=[0.05, 0.95, 0.475],cl_decay_steps=2000,
                 hidden_size=64,points_per_hour=args.points_per_hour)

        elif args.model_name == 'DPG_Mixer':
            args.layers = 4
            self.model = DPG_Mixer(num_nodes=args.num_nodes, in_dim=args.num_features, supports=adj,
                                 out_dim=1, pred_len=args.pred_len, d_model=args.d_model,
                                 d_ff=args.d_ff,
                                 kernel_size=2, layers=args.layers, blocks=1, args=args)

        
        else:
            raise NotImplementedError

    '''Code under one epoch'''
    def train_test_one_epoch(self,args,dataloader,adj,save_manager: tu.save.SaveManager,epoch,mode='train',max_iter=float('inf'),**kargs):
        if mode == 'train':
            self.model.train()
            self.optimizer.zero_grad()
        elif mode == 'test' or mode =='val':
            self.model.eval()
        else:
            raise NotImplementedError

        metric_logger = tu.metric.MetricMeterLogger() 
        # Initialize a dictionary to record the corresponding training loss results.


        for index, unpacked in enumerate(
                metric_logger.log_every(dataloader, header=mode, desc=f'{mode} epoch {epoch}')):
            if index > max_iter:
                break
            seqs, seqs_time,targets,targets_time = unpacked # (B,L,C,N)
            seqs, targets = seqs.cuda().float(), targets.cuda().float()
            seqs_time, targets_time = seqs_time.cuda().float(), targets_time.cuda().float()
            seqs,targets=seqs.permute(0,2,3,1),targets.permute(0,2,3,1)# (B,L,C,N)
            seqs_time, targets_time = seqs_time.permute(0, 2, 3, 1), targets_time.permute(0, 2, 3, 1) #(B,C,N=1,L)
            self.adj = np.array(self.adj)
            pred = self.model(seqs,self.adj,seqs_time=seqs_time,targets_time=targets_time,targets=targets,mode=mode,index=index,epoch=epoch)  # 输入模型
            if (args.model_name=='TESTAM') and mode=='train':
                pred,loss_part=pred[0],pred[1]

            targets = targets[:, 0:1, ...]
            if pred.shape[1]!=1:
                pred = pred[:,0:1,...]

            loss = self.criterion(pred.to(targets.device), targets) 

            # check nan
            if np.isnan(np.array(loss.detach().cpu())).any():
                print('Pred exist nan')
                break

            mse = torch.mean(torch.sum((pred - targets) ** 2, dim=1).detach())
            mae = torch.mean(torch.sum(torch.abs(pred - targets), dim=1).detach())

            metric_logger.update(loss=loss, mse=mse, mae=mae)  # update loss

            step_logs = metric_logger.values()
            step_logs['epoch'] = epoch
            save_manager.save_step_log(mode, **step_logs)  # Save the training loss for each batch

            if mode == 'train':
                if args.model_name=='MegaCRN':
                    loss = loss + loss_part
                loss.backward()
                # Gradient clipping
                if args.clip_max_norm > 0: 
                    nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), args.clip_max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        epoch_logs = metric_logger.get_finish_epoch_logs()
        epoch_logs['epoch'] = epoch
        save_manager.save_epoch_log(mode, **epoch_logs)  # Save the training loss for each epoch

        return epoch_logs

    def train(self):
        args = self.agrs
        if args.resume!=True:
            tu.config.create_output_dir(args)  # Create the output directory
            print('output dir: {}'.format(args.output_dir))
            start_epoch = 0
        else:
            start_epoch=self.start_epoch


        # The following are the saved hyperparameters.
        save_manager = tu.save.SaveManager(args.output_dir, args.model_name, 'mse', compare_type='lt', ckpt_save_freq=30)
        save_manager.save_hparam(args)

        max_iter = float('inf')  

        # The formal training begins now.
        for epoch in range(start_epoch, args.end_epoch):
            if tu.dist.is_dist_avail_and_initialized(): 
                self.train_sampler.set_epoch(epoch)
                self.val_sampler.set_epoch(epoch)
                self.test_sampler.set_epoch(epoch)

            tu.dist.barrier() 

            # train
            self.train_test_one_epoch(args,self.train_dataloader,self.adj, save_manager, epoch, mode='train')

            self.lr_optimizer.step()  # lr decay

            # val
            val_logs = self.train_test_one_epoch(args, self.val_dataloader, self.adj, save_manager, epoch, mode='val')

            # check nan
            if np.isnan(val_logs['mse']).any():
                print('Pred exist nan')
                break
            # test
            test_logs = self.train_test_one_epoch(args,self.test_dataloader,self.adj, save_manager, epoch,mode='test')


            # Early stopping mechanism
            self.early_stopping(val_logs['mse'], model=self.model, epoch=epoch)
            if self.early_stopping.early_stop:
                break
        # Training completed, reading the best weights.
        try:
            dp_mode = args.args.dp_mode
        except AttributeError as e:
            dp_mode = True
        output_path = os.path.join(self.output_path, args.data_name + '_best_model.pkl')
        self.load_best_model(path=output_path, args=args, distributed=dp_mode)


    def ddp_module_replace(self,param_ckpt):
        return {k.replace('module.', ''): v.cpu() for k, v in param_ckpt.items()}

    # TODO Load the best model
    def load_best_model(self, path, args=None, distributed=True):

        ckpt_path = path
        if not os.path.exists(ckpt_path):
            print('The path {0} does not exist, and the model parameters are all randomly initialized.'.format(ckpt_path))
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

        # Read the best weights
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


        # Create a CSV file to record training results
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

        time = datetime.now().strftime('%Y%m%d-%H%M%S')  # Get the current system time
        a_log = [{'dataset': args.data_name, 'model': args.model_name, 'time': time,
                  'LR': args.lr,
                  'batch_size': args.batch_size,
                  'seed': args.seed, 'MAE': mae, 'MSE': mse,'RMSE':rmse,"MAPE":mape,'seq_len': args.seq_len,
                  'pred_len': args.pred_len,'d_model': args.d_model, 'd_ff': args.d_ff,
                  'test_cost_time': test_cost_time,
                  # 'e_layers': args.e_layers, 'd_layers': args.d_layers,
                  'info': args.info,'output_dir':args.output_dir}]
        Write_csv.write_csv_dict(log_path, a_log, 'a+')





