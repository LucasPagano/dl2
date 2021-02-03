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

segmentation_classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void'
]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

mean = [0.485, 0.456, 0.406]
m_mean = [-i for i in mean]
std = [0.229, 0.224, 0.225]
std_inv = [1 / i for i in std]
mean_inv = [i * j for i, j in zip(m_mean, std_inv)]
unorm = transforms.Normalize(mean=mean_inv, std=std_inv)

dataset = Cityscapes('./datasets/Cityscapes/', split='val', mode='fine',
                     target_type='semantic', transform=transform)

data_iter = iter(dataset)
log_nb = 5
cpt = 0
for data, gt in data_iter:
    if cpt < log_nb:
        original_image = np.moveaxis((unorm(data.data) * 255).numpy().astype(np.uint8), 0, -1)
        gt = np.array(gt).astype(np.uint8)

        batch_images = data.to(device)

        out = fcn(batch_images.unsqueeze(0))["out"][0]
        predictions = out.argmax(0).cpu().numpy().astype(np.uint8)

        to_log = {"seg": wb_mask(bg_img=original_image, pred_mask=predictions, true_mask=gt)}
        wandb.log(to_log)
        cpt += 1
print("Done!")
