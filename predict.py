'''Training'''
from scipy.io import loadmat
import numpy as np
import argparse
import configparser
import torch
import json
from utils import printlog,print_signal
from skimage.segmentation import slic,join_segmentations,felzenszwalb
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import scale, minmax_scale
import os
from PIL import Image
# from utils import get_graph_list, split, get_edge_index
import math
from Model.module import Link_Net,SubGcnFeature, GraphNet
from torch_geometric.loader import DataLoader
from Trainer import Trainer
from Monitor import GradMonitor
import random
from visdom import Visdom
from tqdm import tqdm
import os
import h5py
from dataset import SZUTreeDataset
from torch.utils.tensorboard import SummaryWriter

# os.environ['CUDA_VISBLE_DEVICES']= '0,1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRAIN SUBGRAPH')
    parser.add_argument('--name', type=str, default='SZUTree',
                        help='DATASET NAME')
    parser.add_argument('--block', type=int, default=100,
                        help='BLOCK SIZE')
    parser.add_argument('--data_path', type=str, default="/home8/ysx/wlq/spgn/dataset",
                        help='data_path')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='ITERATION')
    parser.add_argument('--num', type=int, default=0.2,
                        help='num of per class')
    parser.add_argument('--gpu', type=int, default=5,
                        help='GPU ID')
    parser.add_argument('--comp', type=int, default=10,
                        help='COMPACTNESS')
    parser.add_argument('--batchsz', type=int, default=32,
                        help='BATCH SIZE')
    parser.add_argument('--run', type=int, default=1,
                        help='EXPERIMENT AMOUNT')
    # parser.add_argument('--spc', type=int, default=0.5,
    #                     help='SAMPLE per CLASS')
    parser.add_argument('--hsz', type=int, default=128,
                        help='HIDDEN SIZE')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='LEARNING RATE')
    parser.add_argument('--wd', type=float, default=0.,
                        help='WEIGHT DECAY')
    arg = parser.parse_args()
    config = configparser.ConfigParser()
    config.read('dataInfo.ini')
    # 从dataInfo.ini读取配置
    dataset_name = arg.name
    band_num = config.getint(dataset_name, 'band')
    # 加上noTree类
    nc = config.getint(dataset_name, 'nc') + 1

    # viz = Visdom(port=17000)
    # Data processing
    # Reading hyperspectral image
    data_path = 'data/{0}/data.mat'.format(arg.name)
    # 清空日志
    log_txt = open("./log.txt", 'w', encoding="utf-8")
    log_txt.close()
    printlog(f"Start loading Dataset",print_signal)
    all_dataset = SZUTreeDataset(arg.data_path,name="SZUTree",data_type="train")
    train_dataset =  SZUTreeDataset(arg.data_path,name="SZUTree",data_type="train",train_num=arg.num,addAbsolutePos=True)
    val_dataset = SZUTreeDataset(arg.data_path,name="SZUTree",data_type="val",train_num=arg.num,addAbsolutePos=True)
    data_num = len(all_dataset)
    index = list(range(data_num))
    random.shuffle(index)
    train_ratio = 0.2
    train_num = int(train_ratio*data_num)
    train_index = index[:train_num]
    val_index = index[train_num:]
    train_dataset = all_dataset[train_index]
    val_dataset = all_dataset[train_index]
    train_loader = DataLoader(train_dataset, batch_size=1)
    val_loader = DataLoader(val_dataset, batch_size=1)
    printlog(f"Finish loading Dataset", print_signal)
   

    for r in range(arg.run):
        print('*'*5 + 'Run {}'.format(r) + '*'*5)
        printlog(f"Run {r}",print_signal)
       
        gcn_link = Link_Net(arg.hsz,nc)

        print("Dataset name:{} nc:{} ".format(arg.name,nc))
        printlog(f"Dataset name:{arg.name} nc:{nc} (Including no tree class)")

        for param in gcn_link.parameters():
            param.requires_grad = False
        subnet = SubGcnFeature(band_num, arg.hsz)
        fullnet = GraphNet(arg.hsz, arg.hsz, nc)
        subnet_pth = "/home8/ysx/wlq/spgn/best_class_epoch437_subnet.pt"
        fullnet_pth = "/home8/ysx/wlq/spgn/best_class_epoch437_fullnet.pt"

    

        subnet.load_state_dict(torch.load(subnet_pth))
        # subnet = DataParallel(subnet)

        fullnet.load_state_dict(torch.load(fullnet_pth))


        gcn_class = [subnet,fullnet]
        optimizer_class = torch.optim.Adam([{'params': gcn_class[0].parameters()},
                                            {'params': gcn_class[1].parameters()}],
                                     weight_decay=arg.wd)
        trainer = Trainer([gcn_class, gcn_link])
        monitor = GradMonitor()
        # 计算模型参数量
        total_params = sum(p.numel() for p in gcn_class[0].parameters()) +  sum(p.numel() for p in gcn_class[1].parameters())
        print("Total number of gcn_class‘s  parameters: ", total_params)

        total_params = sum(p.numel() for p in gcn_link.parameters())
        print("Total number of gcn_link‘s  parameters: ", total_params)
 

        device = torch.device('cuda:{}'.format(arg.gpu)) if arg.gpu != -1 else torch.device('cpu')
        device_ids = [arg.gpu]
        device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
      
        print("device:",device)
        max_class_acc = 0
        max_link_acc = 0
        save_root = 'models/{}/{}/{}'.format(arg.name, arg.block, arg.num)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        pbar = tqdm(range(arg.epoch))
       
        for epoch in pbar:
            seg_class, seg_link = trainer.predict_link(val_loader, epoch,device_ids)
            break

    print('*'*5 + 'FINISH' + '*'*5)

