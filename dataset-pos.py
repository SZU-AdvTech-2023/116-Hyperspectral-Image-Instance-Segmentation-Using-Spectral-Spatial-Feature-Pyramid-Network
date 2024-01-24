
from torch_geometric.data import Dataset
import h5py
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from sklearn.decomposition import PCA
from utils import get_graph_list,get_edge_index,show_label
import os
import copy
from torch_geometric.data import Data, Batch
from skimage.segmentation import slic
from utils import printlog,show_segment,get_rgb,removeNoise,print_signal,getTreeMaskByNDVI,removeNoTree
from tqdm import tqdm
from Model.module import PositionEmbeddingSine
from PIL import ImageFilter
import random
from torch_geometric.transforms import BaseTransform
noise_thresh = 10
train_ratio = 0.8
import glob
from pysnic.algorithms.snic import snic
import skimage.color
from pysnic.algorithms.polygonize import polygonize
from pysnic.algorithms.ramerDouglasPeucker import RamerDouglasPeucker
import math
from sklearn.preprocessing import scale,minmax_scale
def removeNotree(seg,seg_id):
    block_thresh = 50
    seg[seg_id==0]=-1
    index = np.unique(seg)
    new_i = 0
    printlog(f"正在移除非树部分数据",print_signal)
    for i in tqdm(index):
        if i == -1:
            continue
        elif np.sum(seg==i) < block_thresh:
            seg[seg == i] = -1
        else:
            seg[seg==i]=new_i
            new_i += 1
    return seg
class SZUTreeDataset(Dataset):
    def __init__(self, root, name='SZUTree',block=100,compactness=10.0, pre_transform=None,data_type="train",train_num=20,addPos=True):
        super(SZUTreeDataset, self).__init__()
        printlog(f"初始化数据集类", print_signal)
        self.block = block
        self.compactness = compactness
        self.root = root
        self.name = name
        self.filepath = os.path.join(self.root, "data")
        # print(os.listdir(self.filepath))
        self.filenames = os.listdir(self.filepath)
        # print(self.filenames)
        self.pre_transform = pre_transform
        self.save_path = os.path.join(self.root, self.name)
        self.snic_path =  os.path.join(self.root, self.name, "snic")
        if not os.path.exists(self.snic_path):
            printlog(f"创建文件夹({self.snic_path})用于保存SNIC超像素分割可视化结果",print_signal)
            os.makedirs(self.snic_path)
        self.rgb_path =  os.path.join(self.root, self.name, "rgb")
        if not os.path.exists(self.rgb_path):
            printlog(f"创建文件夹({self.rgb_path})用于保存数据的RGB图片",print_signal)
            os.makedirs(self.rgb_path)
        # 判断目录是否存在，不存在则创建

        if not os.path.exists(self.save_path):
            printlog(f"创建文件夹({self.save_path})用于存储处理好的数据",print_signal)
            os.makedirs(self.save_path)
        self.fullGraph_list = []
        self.fullGraph_list_path = os.path.join(self.save_path,"fullGraph.pt")

        if os.path.exists(self.fullGraph_list_path):
            printlog(f"已经有处理好的图，直接加载，加载路径;{self.fullGraph_list_path}",print_signal)
            self.fullGraph_list = torch.load(self.fullGraph_list_path)["fullGraph_list"]
            # 重新生成数据后删除
            # for fullgraph in self.fullGraph_list:
            #     fullgraph.num_nodes = len(fullgraph.subGraph_list)
        else:
            self.process()
        if addPos:
            self.fullGraph_list_addPos_path = self.fullGraph_list_path.replace("fullGraph", "fullGraph_addPos")
            if os.path.exists(self.fullGraph_list_addPos_path):
                self.fullGraph_list = torch.load(self.fullGraph_list_addPos_path)["fullGraph_list"]
            else:
                self.addPos()
                state = {"fullGraph_list": self.fullGraph_list}
                torch.save(state, self.fullGraph_list_addPos_path)
        if data_type == "train":
            self.getTrainLabel(train_num)
        elif data_type == "val":
            self.getValLabel(train_num)
        # 1111
    def _download(self):
        pass
    @property
    def raw_dir(self):
        """原始文件的文件夹"""
        pass
        # return self.filepath

    @property
    def processed_dir(self):
        """处理后文件的文件夹"""
        pass
        # return self.save_path
    @property
    def raw_file_names(self):
        pass
        # return self.filenames

    @property
    def processed_file_names(self) -> str:
        """处理后的文件名，如果在 processed_dir 中找到则跳过 process"""
        return ["fullGraph.pt"]

    def download(self):
        pass

    # 添加位置编码
    def addPos(self):
        posEmbedding = PositionEmbeddingSine(num_pos_feats=64)
        fullgraph = self.fullGraph_list[0]
        data = fullgraph.data
        seg = fullgraph.seg_super_pixel
        subgraph_list = fullgraph.subGraph_list
        output = posEmbedding(data)
        for subgraph in tqdm(subgraph_list):
            subgraph.x = output[seg==subgraph.node_index]

    # 从路径中加载数据
    def load_data(self,filepath,filenames):
        printlog("加载原始数据",print_signal)
        npz_save_path = os.path.join(self.save_path,"npz")
        if not os.path.exists(npz_save_path):
            printlog(f"npz路径不存在，新建路径：{npz_save_path}", print_signal)
            os.makedirs(npz_save_path)
        # 存在则读取返回，不存在则加载数据

        printlog(f"没有处理好的原始数据，从所给文件夹路径加载",print_signal)
        # 只有一张图
        # name = filenames[0]
        name = filenames
        npz_name = name.replace(".mat", ".npz")
        npz_path = os.path.join(npz_save_path, npz_name)
        if os.path.exists(npz_path):
            printlog(f"有预处理好的原始数据，直接加载，加载路径：{npz_path}", print_signal)
            m = np.load(npz_path)
            # data, seg_tree, seg_label, seg_mask = m["data"], m["seg_tree"], m["seg_label"], m["seg_mask"]
            data, seg_tree, seg_label = m["data"], m["seg_tree"], m["seg_label"]
 

        else:

            data_path = os.path.join(filepath, name)
            m = h5py.File(data_path)
            data = m["data"]
            data = np.transpose(data)
            # 处理异常值,把大于均值10倍的数替换为均值
            mean = np.mean(data)
            data[data>10*mean]=mean
            # seg_mask = m["mask"]
            # seg_mask = np.transpose(seg_mask)
            h, w, c = data.shape
            printlog(f"原始数据的维度为：（{h, w, c}）", print_signal)


            # 掩码

            # data = np.multiply(data, np.reshape(seg_mask,(h,w,1)))
            # 归一化

            data = data.reshape((h * w, c))
            data = np.array(data,dtype=np.float32)
            printlog("利用 sklearn 的minmax_scale方法 对当前数据做过一化", print_signal)
            minmax_scale(data,copy=False)
            # data = scale(data).reshape((h, w, c))
            data = data.reshape((h, w, c))

            seg_tree = m["label_id"]
            seg_tree = np.transpose(seg_tree)

            seg_label = m["label_class"]
            seg_label = np.transpose(seg_label).astype(np.int8) -1


            save_dict = {"data": data, "seg_tree": seg_tree, "seg_label": seg_label}
            np.savez(npz_path, **save_dict)

        # return data,seg_tree,seg_label,seg_mask
        return data,seg_tree,seg_label
    # 超像素分割
    def super_pixel(self,data,filename,seg_tree,block=100,compactness=10):
        # block 块大小
        printlog("采用 SNIC算法 对原始数据做超像素分割",print_signal)
        super_pixel_save_path = os.path.join(self.save_path,"super_pixel")

        # 创建目录
        if not os.path.exists(super_pixel_save_path):
            os.makedirs(super_pixel_save_path)


        seg_super_pixel_name = filename.replace(".mat","_super_pixel.npz")

        seg_super_pixel_path = os.path.join(super_pixel_save_path,seg_super_pixel_name)
        h, w, c = data.shape

        # 超像素个数
        n_superpixel = int(math.ceil((h * w) / block))
        # mask = seg_mask
        rgb = data[:, :, [45, 29, 14]]
        img_name = filename.replace(".mat", ".png")
        rgb_save_path = os.path.join(self.rgb_path, "rgb_" + img_name)
        # img = get_rgb(rgb, mask, gamma=0.5, save_path=rgb_save_path)
        img = get_rgb(rgb, gamma=0.5, save_path=rgb_save_path)
        printlog(f"RGB图片保存至路径：{self.rgb_path} ", print_signal)
        # 存在则读取返回，不存在则加载数据
        if os.path.exists(seg_super_pixel_path):

            printlog(f"路径：{seg_super_pixel_path} 下有做好超像素分割数据，直接读取", print_signal)
            save_dict = np.load(seg_super_pixel_path)
            # seg_snic, seg_snic_withnoise, seg_slic = save_dict["seg_snic"], save_dict["seg_snic_withnoise"], save_dict["seg_slic"]
            seg_slic = save_dict["seg_slic"]

        else:
            printlog("没有做好超像素分割的数据，开始超像素分割", print_signal)


            seg_slic = slic(img, n_superpixel, compactness, start_label=0)
            # 删除非树部分
            seg_slic = removeNotree(seg_slic, seg_tree)

            printlog(f"超像素分割结果保存到{seg_super_pixel_path}", print_signal)
            save_dict = {"seg_slic":seg_slic}
            np.savez(seg_super_pixel_path, **save_dict)
        slic_save_path = os.path.join(self.snic_path, f"slic_{n_superpixel}_{compactness}_" + img_name)
        show_segment(img, seg_slic+1, save_path=slic_save_path, dpi=600, figsize=(8, 8))
        printlog(f"超像素分割完成，slic 分割结果保存至路径：{slic_save_path} ", print_signal)




        # return seg_snic,seg_slic
        return seg_slic

    # 处理
    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        printlog(f"路径：{self.processed_file_names} 不存在处理好的数据，开始预处理", print_signal)
        printlog("开始构建图数据集",print_signal)
        # 一个是分类数据集，一个是边连接数据集（同树判断）
        self.fullGraph_list = []


        class_save_path = os.path.join(self.save_path, "class")
        link_save_path = os.path.join(self.save_path,"link")


        if not os.path.exists(class_save_path):
            printlog(f"Class dataset保存路径不存在，创建目录：{class_save_path}",print_signal)
            os.makedirs(class_save_path)
        if not os.path.exists(link_save_path):
            printlog(f"Link dataset保存路径不存在，创建目录：{link_save_path}",print_signal)
            os.makedirs(link_save_path)

        subGraph_save_path = os.path.join(self.save_path, "subGraph")
        if not os.path.exists(subGraph_save_path):
            printlog(f"subGraph保存路径不存在，创建目录：{subGraph_save_path}",print_signal)
            os.makedirs(subGraph_save_path)

        fullGraph_save_path = os.path.join(self.save_path, "fullGraph")
        if not os.path.exists(fullGraph_save_path):
            printlog(f"fullGraph保存路径不存在，创建目录：{fullGraph_save_path}",print_signal)
            os.makedirs(fullGraph_save_path)
        for i in range(len(self.filenames)):
            name = self.filenames[i]
            data, seg_tree, seg_label = self.load_data(self.filepath, name)
            # printlog(f"data shape : {data.shape} seg_tree shape: {seg_tree.shape}, seg_label shape : {seg_tree.shape}",
            #          print_signal=True)
            # seg_snic,seg_slic = self.super_pixel(data, seg_mask,block=self.block,compactness=self.compactness)
            seg_slic = self.super_pixel(data,name,seg_tree, block=self.block, compactness=self.compactness)
            # 选一种超像素分割
            # seg_super_pixel = seg_snic
            seg_super_pixel = seg_slic
            seg_tree = np.array(seg_tree, dtype=np.int16)


            # 有类别标签
            subGraph_name = name.replace(".mat", ".pt")
            subGraph_path = os.path.join(subGraph_save_path,subGraph_name)
            if os.path.exists(subGraph_path):
                printlog(f"{subGraph_name}已存在，直接读取，目录：{subGraph_path}",print_signal)
                state = torch.load(subGraph_path)
                subGraph_list = state["subGraph_list"]
            else:
                printlog(f"{subGraph_name}不存在，创建并保存到:{subGraph_path}",print_signal)
                subGraph_list = get_graph_list(data, seg_super_pixel, seg_label)
                state = {"subGraph_list":subGraph_list}
                torch.save(state,subGraph_path)
            # subGraph = Batch.from_data_list(graph_list)

            printlog(f"正在构获取边", print_signal)


            fullGraph_edge_index_name = name.replace(".mat","_edge_index.npz")
            fullGraph_edge_index_path = os.path.join(fullGraph_save_path,fullGraph_edge_index_name)
            if os.path.exists(fullGraph_edge_index_path):
                printlog(f"{fullGraph_edge_index_name}已存在,直接加载，加载目录:{fullGraph_edge_index_path}",print_signal)
                m = np.load(fullGraph_edge_index_path)
                edge_index,link_labels = m["edge_index"],m["link_labels"]
                # 修正edge_index-1 错误
                # edge_index = edge_index + 1
                # m = {"edge_index": edge_index, "link_labels": link_labels}
                # np.savez(fullGraph_edge_index_path, **m)
            else:
                printlog(f"{fullGraph_edge_index_name}不存在,创建并保存，保存目录:{fullGraph_edge_index_path}",
                         print_signal)
                # 加入单树分割标签
                edge_index, _, link_labels, _ = get_edge_index(seg_super_pixel, seg_tree)
                m = {"edge_index":edge_index,"link_labels":link_labels}
                np.savez(fullGraph_edge_index_path, **m)

            fullGraph = Data(None,
                             edge_index=torch.from_numpy(edge_index) if isinstance(edge_index, np.ndarray) else edge_index,
                             num_nodes=len(subGraph_list),
                             data = torch.from_numpy(data) if isinstance(data, np.ndarray) else data,
                             seg_super_pixel=torch.from_numpy(seg_super_pixel) if isinstance(seg_super_pixel, np.ndarray) else seg_super_pixel,
                             seg_tree=torch.from_numpy(seg_tree) if isinstance(seg_tree, np.ndarray) else seg_tree,
                             seg_label=torch.from_numpy(seg_label) if isinstance(seg_label, np.ndarray) else seg_label,
                             link_labels=torch.from_numpy(link_labels) if isinstance(link_labels, np.ndarray) else link_labels,
                             subGraph_list = subGraph_list)
            self.fullGraph_list.append(fullGraph)
        state = {"fullGraph_list":self.fullGraph_list}
        torch.save(state,self.fullGraph_list_path)


        img = show_label(data, seg=seg_super_pixel, seg_class=seg_label, seg_tree=seg_tree)

    def _process(self):

        pass

    def indices(self):
        return range(len(self.fullGraph_list))
    def len(self) -> int:
        len(self.fullGraph_list)

    def get(self, idx):
        # print(self.graph_list[idx])
        # k = self.k_graph_crop(self.graph_list[idx])
        return self.fullGraph_list[idx]

    def getTrainLabel(self,num_per_class):


        for fullGraph in self.fullGraph_list:
            subgraph = Batch.from_data_list(fullGraph.subGraph_list)
            class_label = np.zeros(subgraph.y.shape) - 1
            y = subgraph.y.numpy()

            # 分类标签
            for i in np.unique(y):
                index = np.nonzero(y == i)[0]
                if num_per_class < 1:
                    printlog(f"num:{num_per_class} < 1， 按百分比划分数据集", print_signal)
                    num = int(len(index) * num_per_class)
                else:
                    num = num_per_class
                np.random.seed(42)
                np.random.shuffle(index)
                if num > len(index):
                    class_label[index] = y[index]
                    print(f"The all number of class {i} is {len(index)}, the train number is {len(index)}")
                else:
                    class_label[index[:num]] = y[index[:num]]
                    print(f"The  all number of class {i} is {len(index)}, the train number is {num}")
            subgraph.y = torch.from_numpy(class_label)
            fullGraph.subGraph_list = subgraph.to_data_list()


            link_label = np.zeros(fullGraph.link_labels.shape) -1
            link = fullGraph.link_labels.numpy()
            index_link,counts = np.unique(link,return_counts=True)
            if num_per_class < 1:
                printlog(f"num:{num_per_class} < 1， 按百分比划分数据集", print_signal)
                num = int((num_per_class+0.3)*np.min(counts))
            for i in index_link:
                if i==1:
                    num = num*2
                index = np.nonzero(link == i)[0]
                np.random.seed(42)
                np.random.shuffle(index)
                if num > len(index):
                    link_label[index] = link[index]
                    print(f"The all number of  link {i} is {len(index)}, the train number is {len(index)}")
                else:
                    link_label[index[:num]] = link[index[:num]]
                    print(f"The all number of link {i} is {len(index)}, the train number is {num}")
            fullGraph.link_labels = torch.from_numpy(link_label)

    def getValLabel(self,num_per_class):


        for fullGraph in self.fullGraph_list:
            subgraph = Batch.from_data_list(fullGraph.subGraph_list)
            class_label = np.zeros(subgraph.y.shape) - 1
            y = subgraph.y.numpy()

            # 分类标签
            for i in np.unique(y):
                index = np.nonzero(y == i)[0]
                if num_per_class < 1:
                    printlog(f"num:{num_per_class} < 1， 按百分比划分数据集", print_signal)
                    num = int(len(index) * num_per_class)
                else:
                    num = num_per_class
                np.random.seed(42)
                np.random.shuffle(index)
                if num > len(index):
                    class_label[index] = y[index]
                    print(f"The all number of class {i} is {len(index)}, the val number is {len(index)}")
                else:
                    class_label[index[num:]] = y[index[num:]]
                    print(f"The  all number of class {i} is {len(index)}, the val number is {len(index)-num}")
            subgraph.y = torch.from_numpy(class_label)
            fullGraph.subGraph_list = subgraph.to_data_list()


            link_label = np.zeros(fullGraph.link_labels.shape) -1
            link = fullGraph.link_labels.numpy()
            index_link,counts = np.unique(link,return_counts=True)
            if num_per_class < 1:
                printlog(f"num:{num_per_class} < 1， 按百分比划分数据集", print_signal)
                num = int((num_per_class+0.3)*np.min(counts))
            for i in index_link:
                if i==1:
                    num = num*2
                index = np.nonzero(link == i)[0]
                np.random.seed(42)
                np.random.shuffle(index)
                if num > len(index):
                    link_label[index] = link[index]
                    print(f"The all number of  link {i} is {len(index)}, the train number is {len(index)}")
                else:
                    link_label[index[num:]] = link[index[num:]]
                    print(f"The all number of link {i} is {len(index)}, the train number is {len(index)-num}")
            fullGraph.link_labels = torch.from_numpy(link_label)
