import torch.utils.data as data
from torch.utils.data import dataset
from dataset.davsod import DAVSODAllPairsCV2, DAVISAllPairsCV2
from torchvision import transforms 
from dataset import custom_transforms as trforms


def get_loader(option, pin_memory=True):
    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(option['trainsize'], option['trainsize'])),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    dataset_DAVSOD = DAVSODAllPairsCV2(option['root'], trainsize=option['trainsize'], sub_set='TrainingSet', transform=composed_transforms_ts)
    # dataset_DAVIS = DAVISAllPairsCV2(root='F:\DAVSOD\TestSet\DAVIS', trainsize=option['trainsize'], sub_set='TrainingSet', transform=composed_transforms_ts)
    data_loader = data.DataLoader(dataset=dataset_DAVSOD,
                                  batch_size=option['batch_size'],
                                  shuffle=True,
                                  num_workers=option['batch_size'],
                                  pin_memory=pin_memory)
    return data_loader
