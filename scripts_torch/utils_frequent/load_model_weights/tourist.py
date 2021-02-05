import torch

net = nanodet()
savePath = ''
def loadnet(net, savePath):

    """整个网络"""
    torch.save(net, savePath)#保存
    netOut = torch.load(savePath)#加载， 这里就是一个字典，自己用的时候在torch里面就可以直接用了

    """参数"""
    torch.save(net.state_dict(), savePath)#save
    weights = torch.load(savePath)#加载参数
    net.load_state_dict(weights)#给自己的模型加载参数

    """保存更多的信息"""
    torch.save({'epoch': 1,
                'state_dict': net.state_dict(),
                'best_loss': 0.5,
                'optimizer': 'optimizer.state_dict()',
                'alpha': 'loss.alpha',
                'gamma': 'loss.gamma'},
                 'filename' + '.pth.tar')
    infos = torch.load(savePath)
    weights = infos['state_dict']
    netOut = net.load_state_dict(weights)


    """可能修改网络, 自己设计的网络减少了某些层举例"""
    weights = net.state_dict()
    infosLoad = torch.load(savePath)
    weightsLoad = infosLoad['state_dict']

    #过滤，自己模型、加载模型 名字相同的键值对， 要
    newDict = {k:v for k,v in weightsLoad.items() if k in weights.keys()}
    weights.update(newDict)
    netOut = net.load_state_dict(weights)

    return netOut