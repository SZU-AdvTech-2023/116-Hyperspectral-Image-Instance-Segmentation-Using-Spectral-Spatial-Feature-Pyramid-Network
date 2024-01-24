'''Training'''
import copy

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

from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
from utils import show_segment,get_rgb
from Trainer import Trainer
from Monitor import GradMonitor
import random
from visdom import Visdom
from tqdm import tqdm
import os
from torch_geometric.nn import DataParallel
import h5py
from Model.module import SubGcnFeature, GraphNet
from dataset import SZUTreeTestDataset
# os.environ['CUDA_VISBLE_DEVICES']= '0,1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from utils import drawresult,getTreeMaskByNDVI,show_mask
def prob_to_pred(probabilities):
    predictions = torch.round(probabilities)
    # print(np.unique(predictions.cpu().detach().numpy(),return_counts=True))
    return predictions

def resetSeg(seg,node_index,edge_index):
    new_seg = np.zeros_like(seg)-1
    new_id = 0
    old_id = np.max(seg)+1
    for i in  tqdm(range(old_id)):
        if i in node_index:
            edge_index[edge_index==i] = new_id
            new_seg[seg == i] = new_id
            new_id += 1
    return new_seg,edge_index
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRAIN SUBGRAPH')
    parser.add_argument('--name', type=str, default='SZUTree',
                        help='DATASET NAME')
    parser.add_argument('--block', type=int, default=100,
                        help='BLOCK SIZE')
    parser.add_argument('--data_path', type=str, default="/home8/ysx/wlq/spgn/dataset",
                        help='data_path')
    parser.add_argument('--epoch', type=int, default=100,
                        help='ITERATION')
    parser.add_argument('--gpu', type=int, default=7,
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
    # 加一类非树类
    nc = config.getint(dataset_name, 'nc')+1

    test_dataset = SZUTreeTestDataset(arg.data_path, name="SZUTree")
    test_loader = DataLoader(test_dataset, batch_size=1)
    # 加载模型
    device = torch.device('cudaz:{}'.format(arg.gpu)) if arg.gpu != -1 else torch.device('cpu')
    subnet_pth = "/home8/ysx/wlq/spgn/best_link_epoch437_subnet.pt"
    fullnet_pth = "/home8/ysx/wlq/spgn/best_link_epoch437_fullnet.pt"

    subnet = SubGcnFeature(band_num, arg.hsz)
    fullnet = GraphNet(arg.hsz, arg.hsz, nc)


    subnet.load_state_dict(torch.load(subnet_pth))
    # subnet = DataParallel(subnet)

    fullnet.load_state_dict(torch.load(fullnet_pth))


    # 模型数据加载到对应设备
    subnet.to(device)
    fullnet.to(device)
    # subgraph = subgraph.to(device)
    for i,batch in enumerate(test_loader):
        fullgraph = batch.to(device)

        subGraph_list = batch.subGraph_list
        subg = Batch.from_data_list(subGraph_list[0])
        # 预测
        with torch.no_grad():
            # subgraph = Batch.from_data_list(fullgraph)
            featrues = subnet(subg)
            node_index = subg.node_index
            graph = batch.to_data_list()[0]
            graph.x = featrues

            logits_class = fullnet.predict_class(graph)

            pred_class = torch.argmax(logits_class.cpu(), dim=-1)
            subset = torch.nonzero(pred_class)
            ### 获取子图，并生成子图位置编码， 变edge_index   节点 x 节点数 num_nodes ,seg_super_pixel
            tree_edge_index,_ = subgraph(subset, graph.edge_index) # 只要类别为树的子图

            graph.x = torch.squeeze(featrues[subset,:])
            graph.num_nodes = len(subset)
            # graph.edge_index = tree_edge_index
            graph.seg_super_pixel,graph.edge_index  = resetSeg(graph.seg_super_pixel.cpu().numpy(),subset.numpy(),tree_edge_index.cpu().numpy())
            graph.seg_super_pixel = torch.from_numpy(graph.seg_super_pixel)
            graph.edge_index  = torch.from_numpy(graph.edge_index)
            data = graph.data.cpu().numpy()
            rgb = data[:, :, [45, 29, 14]]
            img = get_rgb(rgb, gamma=0.5)
            show_segment(rgb, graph.seg_super_pixel.numpy()+1, save_path=None, dpi=600, figsize=(8, 8), color=(1, 1, 0), show=True)
            test_dataset.addAbsolutePos(graph)  ## 添加子图位置编码
            graph = graph.to(device)

            prob_link = fullnet.predict_link(graph)

            pred_class = torch.argmax(logits_class, dim=-1).cpu().numpy()
            pred_class = pred_class[subset].reshape(-1)
            pred_link = prob_to_pred(prob_link).cpu().numpy()


            seg = graph.seg_super_pixel.cpu().numpy()

            # seg_tree = graph.seg_tree.cpu().numpy()
            # seg_class = np.zeros(seg.shape,dtype=int)-1
            edge_index = graph.edge_index.cpu().numpy()

            node_index = np.arange(graph.num_nodes)



            pred_link = torch.round(prob_link)
            data = graph.data.cpu().numpy()
            # drawresult(img,seg_snic,edge_index,pred_class.detach().cpu().numpy(),pred_link.detach().cpu().numpy())
            seg_class, seg_link = drawresult(data, seg, node_index, edge_index, pred_class, pred_link,
                                             save_name=f"{i}_pred_result_")
    print('*' * 5 + 'FINISH' + '*' * 5)
