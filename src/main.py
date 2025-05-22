import dataloader
import os  
import argparse
import random
import numpy as np
import torch as th
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UniST_model


def setup_init(seed):  
    #random.seed()	固定 Python 内置随机函数
    #np.random.seed()	固定 Numpy 的随机性
    #torch.manual_seed()	固定 PyTorch 的 CPU 随机性
    #torch.cuda.manual_seed()	固定 GPU 上的随机性
    #PYTHONHASHSEED	固定 Python 的哈希值顺序（间接影响迭代顺序）
    #cudnn.deterministic=True	让 CUDA 使用确定性算法（但会慢）
    #udnn.benchmark=False	禁用优化搜索路径，进一步保证结果稳定
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

def create_argparser():
    defaults = dict(
        # experimental settings
        task = 'short',
        dataset = 'Crowd',
        mode='training', # ['training','prompting','testing']
        file_load_path = '',
        used_data = '',
        process_name = 'process_name',
        prompt_ST = 0,
        his_len = 6,
        pred_len = 6,
        few_ratio = 0.5,
        stage = 0,

        # model settings
        mask_ratio = 0.5,
        patch_size = 2,
        t_patch_size = 2,
        size = 'middle',
        no_qkv_bias = 0,
        pos_emb = 'SinCos',
        num_memory_spatial = 512,
        num_memory_temporal = 512,
        conv_num = 3,
        prompt_content = 's_p_c',

        # pretrain settings
        random=True,
        mask_strategy = 'random', # ['random','causal','frame','tube']
        mask_strategy_random = 'batch', # ['none','batch']
        
        # training parameters
        lr=1e-3,
        min_lr = 1e-5,
        early_stop = 5,
        weight_decay=1e-6,
        batch_size=256,
        log_interval=20,
        total_epoches = 200,
        device_id='0',
        machine = 'machine_name',
        clip_grad = 0.05,
        lr_anneal_steps = 200,
        batch_size_1 = 64,
        batch_size_2 = 32,
        batch_size_3 = 16,
    )
    parser = argparse.ArgumentParser()
    return parser

def main():
    # Load the dataset
    args = create_argparser().parse_args()
    
    data = dataloader.data_load('./dataset/Multi_regional_MeteoSat/MeteoSat_AF_2022.npy')
    
    #固定一个种子100
    setup_init(100)
    
    device = dev(args.device_id)
    #创建一个 UniST_model 模型实例，传入配置参数（如大小、patch维度等）
    model = UniST_model(args=args).to(device)
    
    
    
    print(args)
    
if __name__ == "__main__":
    main()
    