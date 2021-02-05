import warnings
warnings.filterwarnings("ignore")
import torch

from model import Net
from dataset_loader import ListDataset, ListDatasetTest
from torch.utils.data import DataLoader
from focal_loss import FocalLoss

if __name__ == '__main__':#必须在这里执行训练，否则会有异常告警
    # 准备数据
    # ********************************************************
    train_data = ListDataset()
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=True,  # 如果机器计算能力好的话，就可以设置为True，
    )
    test_data = ListDatasetTest()
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size = 32,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    # -------------------------------------------------------

    # 训练配置
    # **********************************************************
    network = Net()
    # netHg = nn.DataParallel(network, devices = [0, 1, 2]) # 并行训练
    # network.load_state_dict(torch.load('model/cifar10_18000.pt'))
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
    # torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    network.to(device)
    # ---------------------------------------------------------

    for epoch in range(100):
        batch_iter = 0
        for imgs, labels in train_loader:
            # 预测
            imgs = imgs.to(device).float() / 255.0
            labels = labels.to(device).float()
            preds = network(imgs).reshape((-1, 10))
            #with torch.no_grad():
            loss_ = FocalLoss(alpha = [1,1,1,1,1,1,1,1,1,1], num_classes = 10)
            loss = loss_(preds, labels)
            if batch_iter % 100==0:
                print('epoch:', epoch, 'batch_iter:',batch_iter,'loss: ',loss.detach().cpu().numpy())
            # 声明optimizer 不被反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_iter += 1
        if epoch % 5 == 0:
            torch.save(network.state_dict(), '/media/q/deep/me/model/' + 'me_cfar10' + '_' + str(epoch) + '.pt')

        if epoch % 1==0:
            right_num = 0.0
            batch_size = 32
            total_num = len(test_loader)*batch_size #这里需要制定以下batch size
            for imgs, labels in test_loader:
                with torch.no_grad():
                    imgs = imgs.to(device).float() / 255.0
                    labels = labels.to(device).int()
                    preds = network(imgs).reshape((-1, 10))
                    loss_ = FocalLoss(alpha=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], num_classes=10)
                    loss = loss_(preds, labels)
                    acc_ = torch.argmax(preds, dim=1)
                    num = torch.sum(acc_==labels)
                    right_num += num
            acc = right_num/total_num
            print('epoch num: ',epoch, 'loss:', loss.detach().cpu().numpy(), "acc: ", acc.detach().cpu().numpy(),
                  'make sure batch size:', batch_size)

        for p in optimizer.param_groups:
            p['lr'] *= 0.95

