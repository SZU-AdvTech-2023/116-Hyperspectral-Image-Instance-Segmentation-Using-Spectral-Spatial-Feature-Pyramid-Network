import json
import matplotlib.pyplot as plt
def read_txtlog(path='/home8/ysx/wlq/spgn/train_state.txt'):
    file = open(path, 'r')
    lines = file.readlines()
    loss_train = []
    loss_test = []
    epoch = []
    class_acc_train = []
    link_acc_train = []
    class_acc_test = []
    link_acc_test = []
    for line in lines:
        dic = json.loads(line)

        loss_train.append(dic["loss_train_link"])
        loss_test.append(dic["loss_test_link"])
        class_acc_train.append(dic["class_acc_train"])
        link_acc_train.append(dic["link_acc_train"])
        class_acc_test.append(dic["class_acc_test"])
        link_acc_test.append(dic["link_acc_test"])

        epoch.append(dic["epoch"])
    plt.figure(num=3, figsize=(5, 4))
    plt.plot(epoch, link_acc_train, color='green', linewidth=1.0, linestyle=':')

    plt.plot(epoch, link_acc_test, color='red', linewidth=1.0, linestyle='-')
    plt.show()
    file.close()

    return epoch,loss_train,loss_test
def plot_loss():


    path_spgn = '/home8/ysx/wlq/spgn/train_state.txt'


    plt.figure(dpi=600)
    epoch,loss_train,loss_test = read_txtlog(path_spgn)

    plt.plot(epoch[::10],loss_train[::10],color='r',linewidth=1.0,linestyle='-')
    plt.plot(epoch[::10], loss_test[::10], color='r', linewidth=1.0, linestyle=':')

    plt.show()
    plt.savefig('./figure/loss_link_16.png')





if __name__ == "__main__":
    plot_loss()

