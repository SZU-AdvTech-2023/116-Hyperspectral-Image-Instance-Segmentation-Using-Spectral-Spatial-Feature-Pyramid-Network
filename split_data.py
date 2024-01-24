

from utils import printlog
import numpy as np
import os
import h5py
from sklearn.preprocessing import MinMaxScaler as scaler
import matplotlib.pyplot as plt
import scipy.io as sio
from tqdm import tqdm
from copy import deepcopy
print_signal = True
cn = 19
train_ratio = 0.8
printlog("开始分割数据", print_signal)


save_path = "/home8/ysx/wlq/spgn/dataset/data/"

raw_path = "/home8/ysx/wlq/spgn/dataset/data/canghai.mat"
printlog(f"加载原始数据：{raw_path}", print_signal)

m = h5py.File(raw_path)
data = m["data"]
data = np.transpose(data)
h, w, c = data.shape

label_id = m["label_id"]
label_id = np.transpose(label_id)

label_class = m["label_class"]
label_class = np.transpose(label_class)
print(f"data: {data.shape} label_id: {label_id.shape} label_class: {label_class.shape}")

for i in  range(h//500+1):
    row_min = i*500
    if i == h//500:
        row_max = -1
    else:
        row_max = (i+1)*500
    for j in range(w//500+1):
        col_min = j*500
        if j == w // 500:
            col_max = -1
        else:
            col_max = (j + 1) * 500
        subgraph_path = os.path.join(save_path,f"data{i}_{j}.mat")
        f = h5py.File(subgraph_path, 'w')  # 写入文件
        subgraph_data = data[row_min:row_max,col_min:col_max,:]
        subgraph_id = label_id[row_min:row_max,col_min:col_max]
        subgraph_class = label_class[row_min:row_max,col_min:col_max]
        f['data'] = np.transpose(subgraph_data)
        f['label_id'] = np.transpose(subgraph_id)
        f['label_class'] = np.transpose(subgraph_class)
        f.close()  # 关闭文件
        print(f"subgraph_data: {subgraph_data.shape} subgraph_id: {subgraph_id.shape} subgraph_class: {subgraph_class.shape}")
        rgb = subgraph_data[:, :, [45, 29, 14]]
        min_max_scaler = scaler(feature_range=(0, 1))
        sub_h,sub_w,_ = rgb.shape
        rgb = rgb.reshape((sub_h * sub_w, 3))
        img = (np.uint8(min_max_scaler.fit_transform(rgb) * 255)).reshape(sub_h, sub_w, -1)
        plt.figure(dpi=600)
        plt.imshow(img)
        plt.axis('off')  # 不显示坐标轴
        plt.show()





