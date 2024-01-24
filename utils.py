"""
Raw image -> Superpixel segmentation -> graph
"""
import numpy as np
import torch
import cv2 as cv
from torch_scatter import scatter
from torch_geometric.data import Data
import copy
from torch import nn
from tqdm import tqdm
import time
print_signal = False

from sklearn.preprocessing import MinMaxScaler as scaler
from skimage import exposure
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

def show_mask(mask):
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.show()
# 通过ndvi获取非树掩膜
def getTreeMaskByNDVI(data,thresh):

    nir = data[:, :, 75]
    red = data[:, :, 51]
    nir[nir==0]=np.nan
    red[red==0]=np.nan

    # 计算NDVI
    ndvi = (nir - red) / (nir + red)
    mask = np.zeros_like(ndvi)
    mask[~np.isnan(ndvi) & (ndvi > thresh)] = 1
    return mask
# 掩码非树部分
def removeNoTree(seg,tree_mask):
    seg += 1
    seg = np.multiply(seg,tree_mask)
    index = np.unique(seg)
    new_id = 1
    printlog("重新编排图序号",print_signal)
    for i in tqdm(index):

        if i == 0:
            continue
        seg[seg==i]=new_id
        new_id += 1
    return np.array(seg,dtype=np.int)

def get_rgb(rgb,gamma=0.5,save_path=None):
    img = copy.deepcopy(rgb)
    h,w,c = img.shape
    img = img.reshape((h * w, c))
    min_max_scaler = scaler(feature_range=(0, 1))
    img = (np.uint8(min_max_scaler.fit_transform(img) * 255)).reshape(h, w, -1)
    # 把mask的地方变白色
    img[np.nonzero(img==0)]=255

    if gamma:
        img = exposure.adjust_gamma(img, gamma=gamma)
    if save_path:
        cv.imwrite(save_path,img[:,:,[2,1,0]])
    return img
def show_segment(img, seg_slic, save_path=None, dpi=600, figsize=(8, 8),color=(1,1,0),show=True):
    res_img = mark_boundaries(img, seg_slic,color=color)
    if save_path:
        cv.imwrite(save_path,res_img[:,:,[2,1,0]]*255)
        print("*" * 10, f"结果已经保存至{save_path}", "*" * 10)
    if show:
        plt.figure(dpi=dpi, figsize=figsize)
        plt.imshow(res_img)
        plt.axis('off')  # 不显示坐标轴
        plt.show()
    return res_img


cm = [[145, 175, 76],
      [221, 186, 198],
      [163, 132, 163],
      [56, 66, 61],
      [71, 99, 112],
      [196, 45, 198],
      [249, 191, 0],
      [66, 114, 196],
      [0, 173, 234],
      [132, 109, 221],
      [252, 158, 193],
      [216, 130, 221],[221, 216, 107],[239, 226, 216],[127, 255, 255],[211, 68, 112],[66, 38, 119], [51, 175, 71],[224, 43, 142]]
cm = np.array(cm)
# 显示标签
def show_label(data,seg,seg_class,seg_tree,color_map=cm, dpi=600, figsize=(8, 8),save_name = "",show=True):
    rgb = data[:, :, [45, 29, 14]]
    # rgb_save_path = os.path.join(self.rgb_path, "rgb_" + img_name)
    # img = get_rgb(rgb, mask, gamma=0.5, save_path=rgb_save_path)
    img = get_rgb(rgb, gamma=0.5)
    class_index = np.unique(seg_class)
    for i in class_index:
        # if i == -1:
        #     continue
        if i == 0:
            continue
        x_index,y_index = np.nonzero(seg_class==i)
        for x,y in zip(x_index,y_index):
            img[x,y,:] = color_map[i,:]
    org_label = show_segment(img, seg_tree, save_path=f"/home8/ysx/wlq/spgn/dataset/SZUTree/rgb/{save_name}org_label.png",
                           color=(0, 1, 0), show=show)
    img = show_segment(img,seg,save_path=f"/home8/ysx/wlq/spgn/dataset/SZUTree/rgb/{save_name}seg.png", color=(1, 1, 0),show=show)
    res_img = show_segment(img, seg_tree, save_path=f"/home8/ysx/wlq/spgn/dataset/SZUTree/rgb/{save_name}seg_tree.png" ,color=(0, 1, 0),show=show)

    return res_img




# 去除超像素噪声
def removeNoise(segment, noise_thresh):
    seg = copy.deepcopy(segment)
    height,width = seg.shape
    seg_id, id_counts = np.unique(seg,return_counts=True)
    noise_id = seg_id[id_counts<=noise_thresh]
    for i in tqdm(noise_id):
        printlog(f"当前超像素id：{i}",print_signal)
        (y_index,x_index) = np.nonzero(seg==i)
        # 可能在去除其他噪声时顺带去除了
        if len(y_index)==0:
            printlog(f"已经去除，忽略跳过", print_signal)
            continue
        # printlog(f"{y_index,x_index}",print_signal)
#         矩形滤波
#         修复越界bug

        if np.min(x_index)>=1:
            x_min = np.min(x_index) - 1
        else:
            x_min = np.min(x_index)
        if np.max(x_index)< width:
            x_max = np.max(x_index) + 1
        else:
            x_min = np.max(x_index)
        if np.min(y_index)>=1:
            y_min = np.min(y_index) - 1
        else:
            y_min = np.min(y_index)
        if np.max(y_index)< height:
            y_max = np.max(y_index) + 1
        else:
            y_max = np.max(y_index)
        filter_box = seg[y_min:y_max,x_min:x_max]
        # 获取滤波器内id情况
        filter_id, filter_counts = np.unique(filter_box[filter_box!=i],return_counts=True)
        new_seg_id = filter_id[filter_counts==np.max(filter_counts)]
        if len(new_seg_id)==1:
            printlog(f"被{new_seg_id}号超像素包围",print_signal)
            seg[seg==i] = new_seg_id
        else:
            printlog(f"被多个超像素包围，分别是：{new_seg_id},选{new_seg_id[0]}修正", print_signal)
            seg[seg==i] = new_seg_id[0]
        printlog(f"把{i}修正为{new_seg_id}",print_signal)
    # 重新编号图节点序号
    printlog(f"重新编号图节点序号", print_signal)
    new_id = 0
    for index in tqdm(np.unique(seg)):
        seg[seg==index] = new_id
        new_id += 1
    return seg
# Constructing graphs by shifting
def get_grid_adj(grid):
    edge_index = list()
    # 上偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[:-1] = grid[1:]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    # 下偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[1:] = grid[:-1]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    # 左偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[:, :-1] = grid[:, 1:]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    # 右偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[:, 1:] = grid[:, :-1]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    return edge_index

# Getting graph list
def get_graph_list(hsi_data, seg, label_seg):
    data = copy.deepcopy(hsi_data)
    graph_node_feature = []
    graph_node_index = []
    graph_edge_index = []
    graph_gt = []
    # print("*" * 10, "开始构建子图", "*" * 10)
    printlog(f"开始构建子图, 标签：{np.unique(seg)}",print_signal)
    for i in tqdm(np.unique(seg)):

        if i == -1:
            continue

        gt,_ = Counter(label_seg[seg == i].reshape(-1)).most_common(1)[0]

        # if gt == -1:
        #     continue
        if np.isnan(np.max(data[seg == i])):
            print("出现问题")
        # 获取节点特征
        graph_node_feature.append(data[seg == i])
        graph_node_index.append(i)
        # 获取邻接信息
        x, y = np.nonzero(seg == i)
        n = len(x)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        grid = np.full((x_max - x_min + 1, y_max - y_min + 1), -1, dtype=np.int32)
        x_hat, y_hat = x - x_min, y - y_min
        grid[x_hat, y_hat] = np.arange(n)
        graph_edge_index.append(get_grid_adj(grid))

        graph_gt.append(gt)

    graph_list = []
    # 数据变换
    for node, edge_index, gt, node_index in zip(graph_node_feature, graph_edge_index, graph_gt,graph_node_index):
        node = torch.tensor(node, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        graph_list.append(Data(x=node, edge_index=edge_index, y=torch.tensor(gt, dtype=torch.long), node_index=node_index))
    return graph_list
   
from collections import Counter
def get_link_labels(seg, seg_tree, source, target):
    # 判断seg中  source, target索引对应的 块是否 在seg_tree 中属于同一块（大部分是），如果是则为同一颗树
    tree_A,_ = Counter(seg_tree[seg==source].reshape(-1)).most_common(1)[0]
    tree_B,_ = Counter(seg_tree[seg==target].reshape(-1)).most_common(1)[0]

    # 同一颗树
    if tree_A == tree_B and tree_A!=0:
        link_label = 1
    else:
        link_label = 0
    return link_label

# Getting adjacent relationship among nodes
def get_edge_index(segment, seg_tree):
    if isinstance(segment, torch.Tensor):
        segment = segment.numpy()

    img = segment.astype(np.int16)  
    kernel = np.ones((3,3), np.int16) 
    expansion = cv.dilate(img, kernel)
    mask = segment == expansion 
    mask = np.invert(mask) 
    # 去掉-1区域
    mask[img==-1]=False
    # 构图
    h, w = segment.shape
    # 用集合去重
    edge_index = set()
    directions = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    indices = list(zip(*np.nonzero(mask)))
    link_labels = []
    for x, y in tqdm(indices):
        for dx, dy in directions:
            adj_x, adj_y = x + dx, y + dy
            if -1 < adj_x < h and -1 < adj_y < w:
                source, target = segment[x, y], segment[adj_x, adj_y]
                # 增加source!=0 and target != 0 去掉seg_tree==0的非数据部分
                if source ==-1 or target == -1:
                    continue
                label = get_link_labels(segment, seg_tree, source, target)
                if source ==0 or target == 0:
                    label = -1
                edge_index.add((source, target, label))
                edge_index.add((target, source, label))
    edge_index = np.array(list(edge_index))
    # link_labels = edge_index[:, 2]
    # 索引从0开始编排
    link_labels = edge_index[:, 2]
    # edge_index = edge_index[:,0:2] - 1
    edge_index = edge_index[:,0:2]

    return torch.tensor(edge_index, dtype=torch.long).T, edge_index, torch.tensor(link_labels, dtype=torch.long), link_labels

def printlog(tips,print_signal=True):
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if print_signal:
        print(f"----------{time_str}----------")
        print(f"**********{tips}**********")
    else:
        log_txt = open("./log.txt", 'a', encoding="utf-8")
        print(f"------{time_str}-----", file=log_txt)
        print(f"*******{tips}******",file=log_txt)
        log_txt.close()
from PIL import Image
def drawresult(data,seg,node_index,edge_index,pred_class,pred_link,save_name,show=True):
    rgb = data[:, :, [45, 29, 14]]

    img = get_rgb(rgb, gamma=0.5)
    h,w = seg.shape
    seg_class = np.zeros(shape=(h,w),dtype=int)
    seg_link = np.zeros(shape=(h,w),dtype=int)



    # 分类结果
    for i in np.unique(pred_class):

        index = node_index[np.nonzero(pred_class==i)]
        for id in index:
            seg_class[seg==id]=i


    # 聚合树结果
    is_visited = np.zeros_like(node_index) - 1 
    for edge_index1,edge_index2,pred_link in zip(edge_index[0,:],edge_index[1,:],pred_link):
        if pred_link==0:
            if is_visited[edge_index1] == -1:
                if np.sum(seg_class[seg == edge_index1]) != 0:
                    is_visited[edge_index1]=edge_index1
                    seg_link[seg == edge_index1] = edge_index1

            if is_visited[edge_index2] == -1:
                if np.sum(seg_class[seg == edge_index2]) != 0:
                    is_visited[edge_index2] = edge_index2
                    seg_link[seg == edge_index2] = edge_index2

        # 大的并到小的
        else:
            min_index = min(edge_index1,edge_index2)
            max_index = max(edge_index1,edge_index2)
            if is_visited[min_index] == -1 :
                is_visited[edge_index1] = min_index
                seg_link[seg == edge_index1] = min_index
                is_visited[edge_index2] = min_index
                seg_link[seg == edge_index2] = min_index

            else:
                min_index = is_visited[min_index]
                is_visited[max_index] = min_index
                seg_link[seg == max_index] = min_index
            ## 把seg_link里的max_index改为min_index
            is_visited[is_visited == max_index] = min_index
            seg_link[seg_link == max_index] = min_index
    # 后处理，把聚合成一棵树的类别统一
    print("后处理，把聚合成一棵树的类别统一")
    for i in tqdm(np.unique(seg_link)):
        final_class, _ = Counter(seg_class[seg_link == i].reshape(-1)).most_common(1)[0]
        seg_class[seg_link == i] = final_class
    return seg_class,seg_link




from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


def get_oa_aa_kappa(predictions,labels):
    # 计算OA（Overall Accuracy）
    oa = accuracy_score(labels, predictions)
    print("Overall Accuracy:", oa)

    # 计算AA（Average Accuracy）
    cm = confusion_matrix(labels, predictions)
    aa = sum(cm.diagonal() / cm.sum(axis=1)) / cm.shape[0]
    print("Average Accuracy:", aa)

    # 计算Kappa
    kappa = cohen_kappa_score(labels, predictions)
    print("Kappa:", kappa)

    # 计算各个类别的准确率
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("Class Accuracy:", class_accuracy)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


probs = [0.2, 0.6, 0.8, 0.3, 0.9]
labels = [0, 1, 1, 0, 1]

# 计算ROC曲线的假正率（FPR）和真正率（TPR）
def plot_roc(probs,labels):
    fpr, tpr, thresholds = roc_curve(labels, probs)

    # 计算AUC
    auc_score = auc(fpr, tpr)
    print("AUC:", auc_score)

    # 画ROC曲线
    plt.figure(dpi=600)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.savefig("./roc.png")
    plt.show()