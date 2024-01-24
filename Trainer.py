import random

import torch
from torch_geometric.data import Data, Batch
from torch.optim import optimizer as optimizer_
# from torch_geometric.utils import accuracy
from torch_geometric.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch import nn
import time
from collections import Counter
import numpy as np
import torch.nn.functional as F
from utils import  show_label,get_oa_aa_kappa,plot_roc

from utils import printlog,print_signal,drawresult
def prob_to_pred(probabilities):
    predictions = torch.round(probabilities)
    # print(np.unique(predictions.cpu().detach().numpy(),return_counts=True))
    return predictions
class Trainer(object):
    r'''Joint trainer'''
    def __init__(self, models: list):
        super().__init__()
        self.models = models
        self.class_loss_func = nn.CrossEntropyLoss()
        self.link_loss_func = nn.BCEWithLogitsLoss()
    def train(self, train_loader: DataLoader, optimizer, device,epoch,epoches, monitor = None, is_l1=True, is_clip=True):
        # intNet = DataParallel(self.models[0])
        # extNet = self.models[1]
        # model = torch.nn.DataParallel(model)
        gcn_class = DataParallel(self.models[0], device_ids=[4, 5, 7])
        gcn_link = self.models[1]
        gcn_class.train()
        gcn_link.train()
        # extNet.train()
        gcn_class.to(device)
        gcn_link.to(device)
        loss_train = 0
        link_acc_train = 0
        class_acc_train = 0
        num_node = 0
        num_link = 0
        num_graph = 0
        # batch_size 为1
        for batch in train_loader:
            batch = batch.to(device)
            subGraph_list =  batch.subGraph_list
            subgraph = Batch.from_data_list(subGraph_list[0])
            label_class = subgraph.y
            # 只分是否是树

            optimizer.zero_grad()  # 梯度清零
            featrues, logits_class = gcn_class(subgraph.to_data_list())

            index = torch.nonzero(label_class!=-1 )

            label_class = torch.squeeze(label_class[index].to(device))
            # print(logits_class.size(), label_class.size())
            logits_class = torch.squeeze(logits_class[index, :])
            # 选取树超像素
            # print(graph.x.shape)
            graph = batch.to_data_list()[0]
            graph.x = featrues


            prob_link = gcn_link(graph)


            label_link = graph.link_labels

            index = torch.nonzero(label_link != -1)
            label_link = label_link[index]
            prob_link = prob_link[index].squeeze(-1)

            loss_class = self.class_loss_func(logits_class,label_class.long())

            loss_link = self.link_loss_func(prob_link, label_link.float())


            loss = loss_link + loss_class

            loss.backward()

            optimizer.step()


            #   计算acc
            pred_class = torch.argmax(logits_class, dim=-1)
            pred_link = prob_to_pred(prob_link)
            num_node += pred_class.shape[0]
            num_link += pred_link.shape[0]
            num_graph += 1
            class_acc_train += (pred_class==label_class).sum().item() # 每个batch正确分类数
            link_acc_train += (pred_link==label_link).sum().item() # 每个batch正确分类数
            printlog(f"Epoch [{epoch}/{epoches}], train loss： {loss/num_graph:.4f}  class loss: {loss_class:.4f}  link loss: {loss_link:.4f} train class acc:{100*class_acc_train/num_node:.4f}% train link acc:{100*link_acc_train/num_link:.4f}%%",print_signal)

        loss_train /= num_graph
        class_acc_train /= num_node
        link_acc_train /= num_link

        return loss_train,loss_class.item(),loss_link.item(),class_acc_train,link_acc_train
    def train_class(self, train_loader: DataLoader, optimizer, device,epoch,epoches, monitor = None, is_l1=True, is_clip=True):

        gcn_class = self.models[0]
        subnet = DataParallel(gcn_class[0], device_ids=[4, 5, 7])
        fullnet: object = gcn_class[1]
        subnet.train()
        fullnet.train()
        subnet.to(device)
        fullnet.to(device)

        loss_train = 0
        # link_acc_train = 0
        class_acc_train = 0
        num_node = 0
        # num_link = 0
        num_graph = 0
        # batch_size 为1
        for batch in train_loader:
            batch = batch.to(device)
            subGraph_list =  batch.subGraph_list
            subgraph = Batch.from_data_list(subGraph_list[0])
            label_class = subgraph.y
            # 只分是否是树

            optimizer.zero_grad()  # 梯度清零
            # featrues, logits_class = gcn_class(subgraph.to_data_list())
            fe = subnet(subgraph.to_data_list())
            graph = batch.to_data_list()[0]
            graph.x = fe

            logits_class = fullnet(graph)

            index = torch.nonzero(label_class!=-1 )

            label_class = torch.squeeze(label_class[index].to(device))
            # print(logits_class.size(), label_class.size())
            logits_class = torch.squeeze(logits_class[index, :])
           
            loss_class = self.class_loss_func(logits_class,label_class.long())


            loss = loss_class

            loss.backward()

            optimizer.step()


            loss_train += loss.item()


            #   计算acc
            pred_class = torch.argmax(logits_class, dim=-1)
            # pred_link = prob_to_pred(prob_link)
            num_node += pred_class.shape[0]
            # num_link += pred_link.shape[0]
            num_graph += 1
            class_acc_train += (pred_class==label_class).sum().item() # 每个batch正确分类数
            # link_acc_train += (pred_link==label_link).sum().item() # 每个batch正确分类数
            printlog(f"Epoch [{epoch}/{epoches}], train loss： {loss/num_graph:.4f}  class loss: {loss_class:.4f}   train class acc:{100*class_acc_train/num_node:.4f}%",print_signal)
            # printlog(f"Epoch [{epoch}/{epoches}], Train loss: {loss_train/num_graph:.4f}  train class acc:{100*class_acc_train/num_node:.4f}% ",print_signal=False)
        # 计算每个epoch
        loss_train /= num_graph
        class_acc_train /= num_node
        # link_acc_train /= num_link

        return loss_train,loss_class.item(),class_acc_train

    def train_link(self, train_loader: DataLoader, optimizer, device_ids,epoch,epoches, monitor = None, is_l1=True, is_clip=True):

        device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
        gcn_class = self.models[0]
        subnet = DataParallel(gcn_class[0], device_ids=device_ids)
        fullnet = gcn_class[1]
        subnet.train()
        fullnet.train()
        subnet.to(device)
        fullnet.to(device)

        loss_train = 0
        link_acc_train = 0
        class_acc_train = 0
        num_node = 0
        num_link = 0
        num_graph = 0

        # batch_size 为1
        for batch in train_loader:
            batch = batch.to(device)
            subGraph_list =  batch.subGraph_list
            subgraph = Batch.from_data_list(subGraph_list[0])
            label_class = subgraph.y
            # 只分是否是树

            optimizer.zero_grad()  # 梯度清零
            # featrues, logits_class = gcn_class(subgraph.to_data_list())
            fe = subnet(subgraph.to_data_list())
            graph = batch.to_data_list()[0]
            graph.x = fe

            logits_class,prob_link = fullnet(graph)

            index = torch.nonzero(label_class!=-1 )

            label_class = torch.squeeze(label_class[index].to(device))
            # print(logits_class.size(), label_class.size())
            logits_class = torch.squeeze(logits_class[index, :])

            label_link = graph.link_labels

            index = torch.nonzero(label_link != -1)
            label_link = label_link[index]
            prob_link = prob_link[index].squeeze(-1)

            loss_class = self.class_loss_func(logits_class,label_class.long())

            loss_link = self.link_loss_func(prob_link, label_link.float())


            loss = loss_link + loss_class

            loss.backward()

            optimizer.step()


            loss_train += loss.item()


            #   计算acc
            pred_class = torch.argmax(logits_class, dim=-1)
            pred_link = prob_to_pred(prob_link)
            num_node += pred_class.shape[0]
            num_link += pred_link.shape[0]
            num_graph += 1
            class_acc_train += (pred_class==label_class).sum().item() # 每个batch正确分类数
            link_acc_train += (pred_link==label_link).sum().item() # 每个batch正确分类数

            printlog(f"Epoch [{epoch}/{epoches}], Train loss: {loss_train/num_graph:.4f} class loss: {loss_class:.4f} link loss: {loss_link:.4f} train class acc:{100*class_acc_train/num_node:.4f}% train link acc:{100*link_acc_train/num_link:.4f}% ",print_signal)
        # 计算每个epoch
        loss_train /= num_graph
        class_acc_train /= num_node
        link_acc_train /= num_link

        return loss_train,loss_class.item(),loss_link.item(),class_acc_train,link_acc_train
    
    def evaluate(self, val_loader, device,epoch,epoches):
        gcn_class = DataParallel(self.models[0], device_ids=[4, 5, 7])
        # gcn_class = DataParallel(self.models[0])
        gcn_link = self.models[1]
        gcn_class.eval()
        gcn_link.eval()
        # extNet.train()
        gcn_class.to(device)
        gcn_link.to(device)
        
        loss_test = 0
        link_acc_test = 0
        class_acc_test = 0
        
        num_node = 0
        num_link = 0
        num_graph = 0
        # batch_size 为1

        for batch in val_loader:
            batch = batch.to(device)
            subGraph_list = batch.subGraph_list
            subgraph = Batch.from_data_list(subGraph_list[0])
            label_class = subgraph.y


            # label_class[label_class > 0] = 1

            featrues, logits_class = gcn_class(subgraph.to_data_list())


            graph = batch.to_data_list()[0]
            graph.x = featrues
            # 修复索引减1问题
            # graph.edge_index = graph.edge_index + 1
            prob_link = gcn_link(graph)

            index = torch.nonzero(label_class != -1)

            label_class = torch.squeeze(label_class[index].to(device))
            # print(logits_class.size(), label_class.size())
            logits_class = torch.squeeze(logits_class[index, :])

            label_link = graph.link_labels
            index = torch.nonzero(label_link != -1)
            label_link = label_link[index]
            prob_link = prob_link[index].squeeze(-1)

            loss_class = self.class_loss_func(logits_class, label_class.long())
            loss_link = self.link_loss_func(prob_link, label_link.float())
            loss = loss_link + loss_class


            loss_test += loss.item()
            # loss_test += loss_class.item()



            #   计算acc
            pred_class = torch.argmax(logits_class, dim=-1)

            pred_link = prob_to_pred(prob_link)
            num_node += pred_class.shape[0]
            num_link += pred_link.shape[0]
            num_graph += 1
          

            class_acc_test += (pred_class == label_class).sum().item()  # 每个batch正确分类数
            link_acc_test += (pred_link == label_link).sum().item()  # 每个batch正确分类数
            printlog(
                f"Epoch [{epoch}/{epoches}], test loss： {loss_test / num_graph:.4f} test class acc:{100 * class_acc_test / num_node:.4f}%",
                print_signal=False)
        # 计算每个epoch
        loss_test /= num_graph
        class_acc_test /= num_node
        link_acc_test /= num_link

        return loss_test,loss_class.item(),loss_link.item(), class_acc_test,link_acc_test
    def evaluate_class(self, val_loader, device,epoch,epoches):
        gcn_class = self.models[0]
        subnet = DataParallel(gcn_class[0], device_ids=[4, 5, 7])
        fullnet = gcn_class[1]
        subnet.train()
        fullnet.train()
        subnet.to(device)
        fullnet.to(device)
       
        loss_test = 0
        link_acc_test = 0
        class_acc_test = 0
       
        num_node = 0
        num_link = 0
        num_graph = 0
        # batch_size 为1

        for batch in val_loader:
            batch = batch.to(device)
            subGraph_list = batch.subGraph_list
            subgraph = Batch.from_data_list(subGraph_list[0])
            label_class = subgraph.y


            # label_class[label_class > 0] = 1

            fe = subnet(subgraph.to_data_list())
            graph = batch.to_data_list()[0]
            graph.x = fe

            logits_class = fullnet(graph)

            index = torch.nonzero(label_class!=-1 )

            label_class = torch.squeeze(label_class[index].to(device))
            # print(logits_class.size(), label_class.size())
            logits_class = torch.squeeze(logits_class[index, :])
          




            loss_class = self.class_loss_func(logits_class, label_class.long())

            # loss = loss_link + loss_class
            loss = loss_class


            loss_test += loss.item()
            # loss_test += loss_class.item()



            #   计算acc
            pred_class = torch.argmax(logits_class, dim=-1)

            # pred_link = prob_to_pred(prob_link)
            num_node += pred_class.shape[0]
            # num_link += pred_link.shape[0]
            num_graph += 1
           

            class_acc_test += (pred_class == label_class).sum().item()  # 每个batch正确分类数
            # link_acc_test += (pred_link == label_link).sum().item()  # 每个batch正确分类数
            printlog(
                f"Epoch [{epoch}/{epoches}], test loss： {loss_test / num_graph:.4f} test class acc:{100 * class_acc_test / num_node:.4f}% ",print_signal)
        # 计算每个epoch
        loss_test /= num_graph
        class_acc_test /= num_node
        link_acc_test /= num_link


        return loss_test,loss_class.item(), class_acc_test

    def evaluate_link(self, val_loader, device_ids, epoch, epoches):
        gcn_class = self.models[0]
        device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
        subnet = DataParallel(gcn_class[0], device_ids=device_ids)
        fullnet = gcn_class[1]
        subnet.train()
        fullnet.train()
        subnet.to(device)
        fullnet.to(device)
       
        loss_test = 0
        link_acc_test = 0
        class_acc_test = 0
      
        num_node = 0
        num_link = 0
        num_graph = 0
        # batch_size 为1
        link_acc = [0, 0]
        link_num_list = [0,0]
        for batch in val_loader:
            batch = batch.to(device)
            subGraph_list = batch.subGraph_list
            subgraph = Batch.from_data_list(subGraph_list[0])
            label_class = subgraph.y

            # label_class[label_class > 0] = 1

            fe = subnet(subgraph.to_data_list())
            graph = batch.to_data_list()[0]
            graph.x = fe

            logits_class, prob_link = fullnet(graph)

            index = torch.nonzero(label_class != -1)

            label_class = torch.squeeze(label_class[index].to(device))
            # print(logits_class.size(), label_class.size())
            logits_class = torch.squeeze(logits_class[index, :])
          
            label_link = graph.link_labels
            index = torch.nonzero(label_link != -1)
            label_link = label_link[index].squeeze(-1)
            prob_link = prob_link[index].squeeze(-1).squeeze(-1)

            loss_class = self.class_loss_func(logits_class, label_class.long())
            loss_link = self.link_loss_func(prob_link, label_link.float())
            loss = loss_link + loss_class
            # loss = loss_class

            loss_test += loss.item()
            # loss_test += loss_class.item()

            #   计算acc
            pred_class = torch.argmax(logits_class, dim=-1)

            pred_link = prob_to_pred(prob_link)
            num_node += pred_class.shape[0]
            num_link += pred_link.shape[0]
            num_graph += 1
           

            class_acc_test += (pred_class == label_class).sum().item()  # 每个batch正确分类数
            link_acc_test += (pred_link == label_link).sum().item()  # 每个batch正确分类数

            for i in range(2):

                index = torch.nonzero(label_link == i)


                pred_l = pred_link[index].squeeze(-1)
                label_l = label_link[index].squeeze(-1)
                link_num_list[i] += label_l.shape[0]
                # print(link_num_list[i])
                link_acc[i] += (pred_l == label_l).sum().item()
                # print(link_acc[i])
            printlog(
                f"Epoch [{epoch}/{epoches}], test loss： {loss_test / num_graph:.4f} test class acc:{100 * class_acc_test / num_node:.4f}%  test link acc:{100 * link_acc_test / num_link:.4f}% test link0 acc:{100 * link_acc[0] /link_num_list[0] :.4f}% test link1 acc:{100 *  link_acc[1]/ link_num_list[1]:.4f}%",
                print_signal=True)
        # 计算每个epoch
        loss_test /= num_graph
        class_acc_test /= num_node
        link_acc_test /= num_link

        return loss_test, loss_class.item(), loss_link.item(), class_acc_test, link_acc_test
    def predict(self, test_loader, device):
        gcn_class = DataParallel(self.models[0], device_ids=[4, 5, 7])

        gcn_link = self.models[1]
        gcn_class.eval()
        gcn_link.eval()

        gcn_class.to(device)
        gcn_link.to(device)

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                subGraph_list = batch.subGraph_list
                subgraph = Batch.from_data_list(subGraph_list[0])
                label_class = subgraph.y
                # 只分是否是树

                featrues, logits_class = gcn_class(subgraph.to_data_list())


                graph = batch.to_data_list()[0]
                graph.x = featrues
                # 修复索引减1问题
                # graph.edge_index = graph.edge_index + 1
                prob_link = gcn_link(graph).squeeze(-1)


                label_link = graph.link_labels

                # loss_test += loss_class.item()

                #   计算acc
                pred_class = torch.argmax(logits_class, dim=-1).cpu().numpy()
                pred_link = prob_to_pred(prob_link).numpy()
                seg = subgraph.seg_super_pixel
                edge_index = subgraph.edge_index.numpy()
                seg_class = np.zeros(seg.shape)
                for i in np.unique(pred_class):
                    if i == 0:
                        continue
                    else:
                        index = np.nonzero(pred_class==i)
                        for id in index:
                            seg_class[seg==id]=i
                index = np.nonzero(pred_link)
                for i in index:
                    if edge_index[i,0] > edge_index[i,1]:
                        seg[seg==edge_index[i, 0]] = edge_index[i,1]
                    else:
                        seg[seg == edge_index[i, 1]] = edge_index[i, 0]
        return pred_class,pred_link
    def predict_link(self, test_loader,epoch,device_ids):
       
        device = device_ids[0]
        gcn_class = self.models[0]
        subnet = DataParallel(gcn_class[0], device_ids=device_ids)
        fullnet = gcn_class[1]
        subnet.train()
        fullnet.train()
        subnet.to(device)
        fullnet.to(device)
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                subGraph_list = batch.subGraph_list
                subgraph = Batch.from_data_list(subGraph_list[0])

                # 只分是否是树
                node_index = subgraph.node_index
                fe = subnet(subgraph.to_data_list())
                graph = batch.to_data_list()[0]
                graph.x = fe

                logits_class, prob_link = fullnet(graph)
                label_class = subgraph.y
                index = torch.nonzero(label_class != -1)



                label_class = torch.squeeze(label_class[index].to(device))
                # print(logits_class.size(), label_class.size())
                logits_class = torch.squeeze(logits_class[index, :])
               
                label_link = graph.link_labels
                index = torch.nonzero(label_link != -1)
                label_link = label_link[index].squeeze(-1)
                prob_link = prob_link[index].squeeze(-1).squeeze(-1)

               
                node_index = np.array(node_index)
              
                pred_class = torch.argmax(logits_class, dim=-1).cpu().numpy()
                get_oa_aa_kappa(pred_class, label_class.cpu().numpy())


                pred_link = prob_to_pred(prob_link).cpu().numpy()
                get_oa_aa_kappa(pred_link, label_link.cpu().numpy())
                plot_roc(prob_link.cpu().numpy(), label_link.cpu().numpy())
                seg = graph.seg_super_pixel.cpu().numpy()
                data = graph.data.cpu().numpy()
                # seg_tree = graph.seg_tree.cpu().numpy()
                # seg_class = np.zeros(seg.shape,dtype=int)-1
                edge_index = graph.edge_index.cpu().numpy()

                # show_label(data, seg, seg_class, seg_tree,save_name=f"predict_{epoch}")
                seg_class,seg_link = drawresult(data, seg, node_index,edge_index, pred_class, pred_link,save_name=f"{epoch}_pred_")
        return seg_class,seg_link
    def predict_class(self, test_loader, device,epoch):
       
        gcn_class = self.models[0]
        subnet = DataParallel(gcn_class[0], device_ids=[4, 5, 7])
        fullnet = gcn_class[1]
        subnet.train()
        fullnet.train()
        subnet.to(device)
        fullnet.to(device)
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                subGraph_list = batch.subGraph_list
                subgraph = Batch.from_data_list(subGraph_list[0])
                label_class = subgraph.y
                # 只分是否是树
                node_index = subgraph.node_index
                fe = subnet(subgraph.to_data_list())
                graph = batch.to_data_list()[0]
                graph.x = fe

                logits_class = fullnet(graph)

                index = torch.nonzero(label_class != -1)

                label_class = torch.squeeze(label_class[index].to(device))
                # print(logits_class.size(), label_class.size())
                logits_class = torch.squeeze(logits_class[index, :])

                index = torch.nonzero(label_class != -1)
                node_index = np.array(node_index)
                node_index = node_index[index.cpu()]
                #   计算acc
                pred_class = torch.argmax(logits_class, dim=-1).cpu().numpy()

                # pred_link = prob_to_pred(prob_link).numpy()
                seg = graph.seg_super_pixel.cpu().numpy()
                data = graph.data.cpu().numpy()
                seg_tree = graph.seg_tree.cpu().numpy()
                seg_class = np.zeros(seg.shape,dtype=int)-1
                for i in np.unique(pred_class):

                    index = node_index[np.nonzero(pred_class==i)]
                    for id in index:
                        seg_class[seg==id]=i
                show_label(data, seg, seg_class, seg_tree,save_name=f"predict_{epoch}")
        return seg_class
    
    #
    def save_model(self, path, name):
        torch.save(self.models[0][0].cpu().state_dict() , name + "_subnet.pt")
        torch.save(self.models[0][1].cpu().state_dict(), name + "_fullnet.pt")




