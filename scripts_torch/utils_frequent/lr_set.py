def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    lr *= (0.1 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr