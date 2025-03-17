# import argparse


# def parse_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', default=42, type=int)
#     parser.add_argument('--skip-first', action='store_true')
#     parser.add_argument('--log-dir', default='./outputs/log_terminal/02-10-nomap-clreps')
#     parser.add_argument('--tb-dir', default='./outputs/log_tensorboard/02-10-nomap-clreps')
#     parser.add_argument('--save-dir', default='')
#     parser.add_argument('--resume', default='')
#     parser.add_argument('--parallel', default='single', choices=['single', 'DP', 'DDP'])
#     parser.add_argument('--device_ids', default='0,1')
#     parser.add_argument("--local_rank", type=int, default=0)
#     parser.add_argument("--world-size", type=int, default=4)
#     parser.add_argument("--amp", action='store_true') 
#     parser.add_argument('--perm-id', default=0, type=str, choices=[str(i) for i in range(5)])
#     parser.add_argument('--dataset', default='ACE', choices=['MAVEN', 'ACE'])
#     parser.add_argument('--stream-root', default='./data_incremental', type=str)
#     parser.add_argument('--max_seqlen', default=120)
#     parser.add_argument('--adamw_eps', default=1e-7)
#     parser.add_argument('--fixed-enum', default=True, type=bool, help="whether to fix the exemplar number")
#     parser.add_argument('--enum', default=1, type=int, help="When 'fixed-num' == False, indicates the the whole memory size\
#                                                             when 'fixed-num' == True, indicates every class's exemplar num")
#     parser.add_argument('--temperature', default=2)
#     parser.add_argument('--task-num', default=5, type=int)
#     parser.add_argument('--early-stop', action='store_true')
#     parser.add_argument('--patience', type=int, default=5)
#     parser.add_argument('--eval_freq', type=int, default=1)

#     parser.add_argument('--my_test', action='store_true')

#     parser.add_argument('--input-map', action='store_true', help="Whether to use input mapping, if False, use span_s to predict trigger type")
#     parser.add_argument('--class-num', type=int, default=10)
#     parser.add_argument('--shot-num', default=5, type=int)
#     parser.add_argument('--e_weight', default=50)
#     parser.add_argument('--no-replay', action='store_true')
#     parser.add_argument('--period', type=int, default=10)
#     parser.add_argument('--epochs', default=20, type=int) 
#     parser.add_argument('--batch-size', default=4, type=int)
#     parser.add_argument('--device', default="cuda:2", help='set device cuda or cpu')
#     parser.add_argument('--log', action='store_true') 
#     parser.add_argument('--log-name', default='temp')
#     parser.add_argument('--data-root', default='./data_incremental', type=str)
#     parser.add_argument('--backbone', default='bert-base-uncased', help='Feature extractor')
#     parser.add_argument('--lr',type=float, default=2e-5)
#     parser.add_argument('--decay', type=float, default=1e-4, help="")
#     parser.add_argument('--no-freeze-bert', action='store_true')
#     parser.add_argument('--dweight_loss', action='store_true')
#     parser.add_argument('--alpha', type=float, default=2.0)
#     parser.add_argument('--beta', type=float, default=3.0)
#     parser.add_argument('--distill', required=True, choices=["fd", "pd", "mul", "none"])
#     parser.add_argument('--rep-aug', required=True, choices=["none", "mean", "relative"])
#     parser.add_argument('--gamma', type=float, default=1)
#     parser.add_argument('--theta',type=float, default=6)
#     # parser.add_argument('--ecl', required=True, choices=["dropout", "shuffle", "RTR", "none"])
#     parser.add_argument('--cl_temp', type=float, default=0.5)
#     parser.add_argument('--ucl', action='store_true')
#     parser.add_argument('--cl-aug', choices=["dropout", "shuffle", "RTR", "none"])
#     parser.add_argument('--sub-max', action='store_true')
#     parser.add_argument('--leave-zero', action='store_true')
#     parser.add_argument('--single-label', action='store_true')
#     parser.add_argument('--aug-repeat-times', type=int, default=1)
#     parser.add_argument('--aug-dropout-times', type=int, default=1)
#     parser.add_argument('--joint-da-loss', default="none", choices=["none", "ce", "dist", "mul"])
#     parser.add_argument('--tlcl', action="store_true")
#     parser.add_argument('--mse-loss', action="store_true")
#     parser.add_argument('--pseudo-label', action="store_true")
#     parser.add_argument('--llm-augment', action="store_true")
#     parser.add_argument('--llm-augment-times', type=int, default=5)
#     parser.add_argument('--sim-event-type', action="store_true")
#     parser.add_argument('--sam', action="store_true")
#     parser.add_argument('--sam-type', type=str, default="current")
#     parser.add_argument('--rho', type=float, default=0.1)
#     parser.add_argument('--skip-first-cl', choices=["ucl", "tlcl", "ucl+tlcl", "none"], default="none")
#     parser.add_argument('--method', type=str)
#     args = parser.parse_args()
#     return args


import os
from moo import *

class Config:
    def __init__(self):
        self.seed = 42
        self.skip_first = False
        self.log_dir = './outputs/log_terminal/02-10-nomap-clreps'
        self.tb_dir = './outputs/log_tensorboard/02-10-nomap-clreps'
        self.save_dir = ''
        self.resume = ''
        self.parallel = 'single'
        self.device_ids = '0,1'
        self.local_rank = 0
        self.world_size = 4
        self.amp = False
        self.perm_id = int(os.environ.get('config.perm_id'))
        self.dataset = 'ACE'
        self.stream_root = './data_incremental'
        self.max_seqlen = 120
        self.adamw_eps = 1e-7
        self.fixed_enum = True
        self.enum = 1
        self.temperature = 2
        self.task_num = 5
        self.early_stop = False
        self.patience = 5 
        self.eval_freq = 1
        self.my_test = False
        self.input_map = False
        self.class_num = 10
        self.shot_num = int(os.environ.get("config.shot_num"))
        self.e_weight = 50
        self.no_replay = False
        self.period = 10
        self.epochs = 30
        self.batch_size = 4
        self.device = "cuda:0"
        self.log = True
        self.log_name = 'temp'
        self.data_root = './data_incremental'
        self.backbone = 'bert-base-uncased'
        self.lr = 2e-5
        self.decay = 1e-4
        self.no_freeze_bert = True
        self.dweight_loss = True
        self.alpha = 2.0
        self.beta = 3.0
        self.distill = 'mul'  # Required field
        self.rep_aug = 'mean'  # Required field
        self.gamma = 1
        self.theta = 6
        self.cl_temp = 0.07
        self.ucl = False
        self.cl_aug = 'shuffle'
        self.sub_max = True
        self.leave_zero = False
        self.single_label = True
        self.aug_repeat_times = 5
        self.aug_dropout_times = 0
        self.joint_da_loss = "none"
        self.tlcl = True
        self.mse_loss = False
        self.pseudo_label = False
        self.llm_augment = False
        self.llm_augment_times = 5
        self.sim_event_type = False
        self.sam = True
        self.sam_type = "current"
        self.rho = 0.05
        self.skip_first_cl = "ucl+tlcl"
        self.method = None
        self.mul_task_type = 'FairGrad'
        self.sam_optimizer = 'LookbehindASAM'