import torch
import wandb
import numpy as np
from torchvision import models, transforms
from torchvision.datasets import Cityscapes


def labels(classes):
    l = {}
    for i, label in enumerate(classes):
        l[i] = label
    return l


def wb_mask(bg_img, pred_mask, true_mask):
    return wandb.Image(bg_img, masks={
        "prediction": {"mask_data": pred_mask, "class_labels": labels(classes_out)},
        "ground truth": {"mask_data": true_mask, "class_labels": labels(classes_gt)}})


## wandb
run = wandb.init(project="wandb-demo")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval().to(device)

classes_out = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

classes_gt = [
    'unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground',
    'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence',
    'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle', 'license plate'
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
mask_list = []
for data, gt in data_iter:
    if cpt < log_nb:
        original_image = np.moveaxis((unorm(data.data) * 255).numpy().astype(np.uint8), 0, -1)
        gt = np.array(gt).astype(np.uint8)

        batch_images = data.to(device)

        out = fcn(batch_images.unsqueeze(0))["out"][0]
        predictions = out.argmax(0).cpu().numpy().astype(np.uint8)
        mask_list.append(wb_mask(bg_img=original_image, pred_mask=predictions, true_mask=gt))
        cpt += 1

wandb.log({"predictions": mask_list})
