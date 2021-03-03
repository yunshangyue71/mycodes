"""
make sure install tensorboardX , pytorch >0.4
"""
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

"""start"""
writer = SummaryWriter() # 这里不放置任何路径， 会自动的按照日期进行

"""add scalar 
loss/lossL2 : lossL2表示一个名字， loss表示这个scalar放到了哪个集合里面，方便检索
"""
writer.add_scalar('loss/lossL2', lossL2,  niter)
writer.add_scalar('loss/lossIou', lossIou, niter)
writer.add_scalars('data/scalar_group',
                   {'xsinx': xsin,
                    'xcosx': xcos,
                    'arctanx': arctanx}, niter)、

"""add image
将一个batch的imgs拼接成一个大的img， 然后再tbx中显示
"""
x = vutils.make_grid(imgs, normalize=True, scale_each=False)
writer.add_image('Image', x, niter)

"""add txt"""
writer.add_text('Text', 'text logged at step:' + str(n_iter), n_iter)

"""add histgram: WB直方图
纵轴 y：每个批次的结果
横轴 x：表示权重取值为x
x y 对应的值：第y批次， 权重在x附近的有多少个
distribution
横轴：第几个批次
纵轴：权重取值为某个数
颜色 深，表示取值为这个数的多

"""
for name, param in network.named_parameters():# 自己命名的层
    writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)
for name, param in network.state_dict().items():
    writer.add_histogram(name, param.clone().cpu().data.numpy(), id)
    #配合network， debug 来进行查阅

"""add graph"""
writer.add_graph(network, imgs, verbose=False)

writer.close()

"""打开"""
tensorboard --logdir runs
打开 localhost:6006

"""add embedding :高纬度的降维显示 对应projector"""
features = imgs.view(100, 784)
writer.add_embedding(features, metadata=label, label_img=imgs.unsqueeze(1))