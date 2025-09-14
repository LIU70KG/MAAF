# 作者：刘成广
# 时间：2024/7/17 下午2:23
# 作者：刘成广
# 时间：2024/7/16 下午10:09
import os
import pickle
import numpy as np
from random import random
from data_loader import get_loader
from solver import Solver
from optuna.visualization import plot_parallel_coordinate
import torch
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn
import optuna
import shap  # 确保已安装 SHAP：pip install shap
import optuna.visualization as vis
# DASS-21 : 抑郁量表≤9分为正常，l0～l3分为轻度，14～20分为中度，21～27分为重度，≥28分为非常严重；


optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}
username = Path.home().name
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir.joinpath('datasets')
data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath('MOSEI'),
             'ur_funny': data_dir.joinpath('UR_FUNNY'), 'daic-woz': data_dir.joinpath('DAIC-WOZ'),
             'cmdc': data_dir.joinpath('CMDC'), 'avec2014': data_dir.joinpath('AVEC2014'),
             'search': data_dir.joinpath('SEARCH')}


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'activation':
                    value = activation_dict[value]
                setattr(self, key, value)

        self.dataset_dir = data_dict[self.data.lower()]
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--best_model_Configuration_Log', type=str, default='./AZVmodel_backbone_report.txt',
                        help='Load the best model to save features')
    # Mode
    parser.add_argument('--mode', type=str, default='train')
    # parser.add_argument('--learningRate_min', type=float, default=1e-6, help='最小学习率，在scheduler设置为Cosine时才生效')  # 1e-6
    parser.add_argument('--learningRate_scheduler_type', type=str, default='Cosine',
                        help='学习率自动减小方式，None为不自动减小')  # 'Cosine' # None

    parser.add_argument('--num_of_class', type=int, default=2, help='分类的类别数量')  # dass21应该是2，ryerson是8
    parser.add_argument('--oneMoLossRatio', type=float, default=0.3, help='单模态损失系数')
    parser.add_argument('--scoreLossRatio', type=float, default=0.3, help='权重损失的系数')

    # Train
    time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    parser.add_argument('--name', type=str, default=f"{time_now}")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epoch', type=int, default=500)  # sota 1000
    parser.add_argument('--patience', type=int, default=50)  # sota 80

    parser.add_argument('--learning_rate', type=float, default=1e-4)  # 1e-4
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)
    # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh"
    parser.add_argument('--activation', type=str, default='relu')
    # Model
    parser.add_argument('--model', type=str,
                        default='AZVmodel', help='one of {SimpleMulT, AZVmodel, TFN, AZVmodel_Mult, LMF, TFN, Simple_concat}')
    parser.add_argument('--data', type=str, default='SEARCH')  # cmdc DAIC-WOZ AVEC2014 SEARCH----------

    parser.add_argument('--needChangeAudioDimInLstm', type=bool, default=False, help='需要在模型中调整音频特征维度')
    parser.add_argument('--CMDC_fold', type=int, default=1, help='交叉验证使用哪一折（1~k），一般手动指定，写自动脚本的时候才用这个参数')
    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    if kwargs.model == "Simple_Fusion_Network":
        kwargs.best_model_Configuration_Log = "./Simple_Fusion_Network_report.txt"
    elif kwargs.model == "AZVmodel":
        kwargs.best_model_Configuration_Log = "./AZVmodel_backbone_report.txt"

    if kwargs.data == "cmdc":
        kwargs.needChangeAudioDimInLstm = False
        kwargs.num_of_class = 2
        kwargs.batch_size = 64
        kwargs.cross_validation = "cmdc_data_all_modal_5"  # 修改进行交叉验证： 1 2 3 4 5
    elif kwargs.data == "DAIC-WOZ":
        kwargs.needChangeAudioDimInLstm = True
        kwargs.num_of_class = 2
        kwargs.batch_size = 64
        kwargs.learning_rate = 1e-4
    elif kwargs.data == "SEARCH":
        kwargs.needChangeAudioDimInLstm = True
        kwargs.num_of_class = 2
        kwargs.batch_size = 64
        kwargs.learning_rate = 1e-4
    elif kwargs.data == "AVEC2014":
        kwargs.needChangeAudioDimInLstm = False
        kwargs.num_of_class = 2
        kwargs.batch_size = 64
        kwargs.learning_rate = 1e-4
    else:
        print("No dataset mentioned")
        exit()
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)



if __name__ == '__main__':

    # Setting random seed
    random_name = str(random())
    random_seed = 3614
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    # Setting the config for each stage
    train_config = get_config(mode='train')
    test_config = get_config(mode='test')
    # Creating pytorch dataloaders
    train_data_loader = get_loader(train_config, shuffle=True)
    test_data_loader = get_loader(test_config, shuffle=False)

    # Solver is a wrapper for model traiing and testing
    solver = Solver
    solver = solver(train_config, test_config, train_data_loader, test_data_loader,is_train=True)


    # Build the model
    solver.build()

    # Train the model (test scores will be returned based on dev performance)
    solver.train()