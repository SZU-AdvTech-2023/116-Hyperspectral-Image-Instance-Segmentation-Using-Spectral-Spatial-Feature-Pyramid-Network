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
    parser.add_argument('--num', type=int, default=0.8,
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


    log_txt = open("./log.txt", 'w', encoding="utf-8")
    log_txt.close()
    printlog(f"Start loading Dataset",print_signal)
    # all_dataset = SZUTreeDataset(arg.data_path,name="SZUTree",data_type="train")
    train_dataset =  SZUTreeDataset(arg.data_path,name="SZUTree",data_type="train",train_num=arg.num,addAbsolutePos=True)
    val_dataset = SZUTreeDataset(arg.data_path,name="SZUTree",data_type="val",train_num=arg.num,addAbsolutePos=True)

    train_loader = DataLoader(train_dataset, batch_size=1)
    val_loader = DataLoader(val_dataset, batch_size=1)
    printlog(f"Finish loading Dataset", print_signal)
 
    for r in range(arg.run):
        print('*'*5 + 'Run {}'.format(r) + '*'*5)
        printlog(f"Run {r}",print_signal)

        gcn_link = Link_Net(arg.hsz,nc)

        print("Dataset name:{} nc:{} ".format(arg.name,nc))
        printlog(f"Dataset name:{arg.name} nc:{nc} (Including no tree class)")

        subnet = SubGcnFeature(band_num, arg.hsz)
        fullnet = GraphNet(arg.hsz, arg.hsz, nc)

        gcn_class = [subnet,fullnet]
        optimizer_class = torch.optim.Adam([{'params': gcn_class[0].parameters()},
                                            {'params': gcn_class[1].parameters()}],
                                           weight_decay=arg.wd)
        trainer = Trainer([gcn_class, gcn_link])
        monitor = GradMonitor()
        # 计算模型参数量
        total_params = sum(p.numel() for p in gcn_class[0].parameters()) + sum(
            p.numel() for p in gcn_class[1].parameters())
        print("Total number of gcn_class‘s  parameters: ", total_params)

        total_params = sum(p.numel() for p in gcn_link.parameters())
        print("Total number of gcn_link‘s  parameters: ", total_params)

        device = torch.device('cuda:{}'.format(arg.gpu)) if arg.gpu != -1 else torch.device('cpu')
        device_ids = [arg.gpu]
        device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
        # model = model.to(device)
        # if args.gpu > 1:
        #     model = torch.nn.DataParallel(model, device_ids=[4, 5, 6, 7])
        print("device:",device)
        max_class_acc = 0
        max_link_acc = 0
        save_root = '/home8/ysx/wlq/SPGN/models/{}/{}/{}'.format(arg.name, arg.block, arg.num)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        pbar = tqdm(range(arg.epoch))
        # Training

        writer = SummaryWriter("/home8/ysx/wlq/SPGN/tensorboard/logs_fina")  # 存放log文件的目录
        for epoch in pbar:
            pbar.set_description_str('Epoch: {}'.format(epoch))

            loss_train, loss_train_class, loss_train_link, class_acc_train, link_acc_train = trainer.train_link(train_loader, optimizer_class, device_ids,epoch,arg.epoch, monitor.clear(), is_l1=True, is_clip=True)
            pbar.set_postfix_str('train loss: {:.4f} class loss: {:.4f} link loss: {:.4f} train class acc:{:.4f}% train link acc:{:.4f}% \n'.format(loss_train,loss_train_class,loss_train_link,100*class_acc_train,100*link_acc_train))
            # pbar.set_postfix_str('train loss: {:.4f} class loss: {:.4f}  train class acc:{:.4f}%'.format(loss_train,loss_train_class,100*class_acc_train))
            loss_test,loss_test_class,loss_test_link, class_acc_test,link_acc_test = trainer.evaluate_link(val_loader, device_ids,epoch,arg.epoch)

            writer.add_scalar('train/loss', loss_train, epoch)
            writer.add_scalar('train/loss_train_class', loss_train_class, epoch)
            writer.add_scalar('train/loss_train_link', loss_train_link, epoch)
            writer.add_scalar('train/class_acc_train', class_acc_train, epoch)
            writer.add_scalar('train/link_acc_train', link_acc_train, epoch)
            writer.add_scalar('valid/loss', loss_test, epoch)
            writer.add_scalar('valid/loss_test_class', loss_test_class, epoch)
            writer.add_scalar('valid/loss_test_link', loss_test_link, epoch)
            writer.add_scalar('valid/class_acc_test', class_acc_test, epoch)
            writer.add_scalar('valid/link_acc_test', link_acc_test, epoch)

            tran_state = {"epoch":epoch,'loss_train': loss_train, 'loss_train_class':loss_train_class,"loss_train_link":loss_train_link,'class_acc_train': class_acc_train, 'link_acc_train': link_acc_train, "loss_test":loss_test,"loss_test_class":loss_test_class,"loss_test_link":loss_test_link,"class_acc_test":class_acc_test,"link_acc_test":link_acc_test}
            with open('./train_state.txt', 'a') as f:
                json.dump(tran_state, f)
                f.write('\n')

            if class_acc_test > max_class_acc:
                max_class_acc = class_acc_test
                if not os.path.exists(save_root):
                    os.makedirs(save_root)
                trainer.save_model(save_root, f"best_class_epoch{epoch}")

            if link_acc_test > max_link_acc:
                max_link_acc = link_acc_test
                if not os.path.exists(save_root):
                    os.makedirs(save_root)
                trainer.save_model(save_root, f"best_link_epoch{epoch}")


        writer.close()
    print('*'*5 + 'FINISH' + '*'*5)

