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
os.environ['CUDA_VISBLE_DEVICES']= '0,1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRAIN SUBGRAPH')
    parser.add_argument('--name', type=str, default='SZUTree',
                        help='DATASET NAME')
    parser.add_argument('--block', type=int, default=100,
                        help='BLOCK SIZE')
    parser.add_argument('--data_path', type=str, default="/home4/ysx/gigcn/dataset",
                        help='data_path')
    parser.add_argument('--epoch', type=int, default=500,
                        help='ITERATION')
    parser.add_argument('--num', type=int, default=0.2,
                        help='num of per class')
    parser.add_argument('--gpu', type=int, default=0,
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


    printlog(f"Start loading Dataset",print_signal)
    # all_dataset = SZUTreeDataset(arg.data_path,name="SZUTree",data_type="train")
    train_dataset =  SZUTreeDataset(arg.data_path,name="SZUTree",data_type="train",train_num=arg.num)
    val_dataset = SZUTreeDataset(arg.data_path,name="SZUTree",data_type="val",train_num=arg.num)

    train_loader = DataLoader(train_dataset, batch_size=1)
    val_loader = DataLoader(val_dataset, batch_size=1)

    fullgraph = train_loader.dataset.fullGraph_list[0]
    subgraph_list = fullgraph.subGraph_list
    # class_y = subgraph_list.y
    mean_node = np.zeros(shape=(6595,112),dtype=float)
    class_node = np.zeros(shape=(6595),dtype=int)

    for i,g in enumerate(subgraph_list):
        mean_node[i] = np.mean(g.x.numpy()[:,:112],axis=0)
        class_node[i] = g.y.numpy().item()
    from scipy import io as sio
    for i in range(1,12):
        class_data = mean_node[np.nonzero(class_node==i),:]
        sio.savemat(f"./class{i}_data.mat",{f"class{i}":class_data})

    printlog(f"Finish loading Dataset", print_signal)
