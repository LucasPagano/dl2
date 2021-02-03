import torch
import wandb
import numpy as np
from torchvision import models, transforms
from torchvision.datasets import Cityscapes


def labels():
    l = {}
    for i, label in enumerate(classes):
        l[i] = label
    return l


def wb_mask(bg_img, pred_mask, true_mask):
    return wandb.Image(bg_img, masks={
        "prediction": {"mask_data": pred_mask, "class_labels": labels()},
        "ground truth": {"mask_data": true_mask, "class_labels": labels()}})


## wandb
run = wandb.init(project="wandb-demo")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval().to(device)

classes = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = Cityscapes('./datasets/Cityscapes/', split='val', mode='fine',
                     target_type='semantic', transform=transform)

data_iter = iter(dataset)
cpt = 0
for data, image in data_iter:
    if cpt < 1:
        batch_images = data.to(device)
        print(batch_images)
        out = fcn(batch_images.unsqueeze(0))
        for key, value in out.items():
            print(key)
            print(value)
            try:
                print(value.size())
            except:
                pass
        to_log = wb_mask(batch_images)
        cpt += 1
