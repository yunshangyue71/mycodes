from yacs.config import CfgNode

#先定义一个根节点,yaml这个文件就是根节点， new_allowed =True:可以在这个里面添加子节点
cfg = CfgNode(new_allowed=True)
cfg.dir = CfgNode(new_allowed=True)
cfg.model = CfgNode(new_allowed=True)
cfg.train = CfgNode(new_allowed=True)
cfg.augment = CfgNode(new_allowed=True)

# cfg.save_dir = '../../script_torch/utils_frequent/'  #根节点的一个子节点，这个子节点就是一个str
# cfg.model = CfgNode()                               #添加一个子节点， 这个子节点，不可以在yaml中添加子节点， 只可以在这个里面书写
# cfg.model.arch = CfgNode(new_allowed=True)          #添加一个子节点，arch 这个子节点，在yaml中可以继续添加子节点

#将yaml中的节点和这里定义的节点进行合并
def load_config(cfg, yamlPath):
    cfg.defrost()
    cfg.merge_from_file(yamlPath)
    cfg.freeze()

if __name__ == '__main__':
    #具体怎么使用
    yamlPath = './config.yaml'
    load_config(cfg, yamlPath)
    #cfg这个就可以使用了
    print(cfg)