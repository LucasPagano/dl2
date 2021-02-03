from typing import Any, Union, List, Optional
import pytorch_lightning as pl

import torch
import torchvision
from torch.utils.data import DataLoader


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        use_cuda = not config.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.args_dloader = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    def setup(self, stage: Optional[str] = None):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        dataset = torchvision.datasets.MNIST("./models", train=True, download=True, transform=transform)
        self.train_set, self.val_set = torch.utils.data.random_split(dataset, [50000, 10000])
        self.test_set = torchvision.datasets.MNIST("./models", train=False, download=True, transform=transform)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return torch.utils.data.DataLoader(
            self.train_set, batch_size=self.config.batch_size, shuffle=True, **self.args_dloader
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return torch.utils.data.DataLoader(
            self.val_set, batch_size=self.config.batch_size, shuffle=False, **self.args_dloader
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return torch.utils.data.DataLoader(
            self.test_set, batch_size=self.config.batch_size, shuffle=False, **self.args_dloader
        )

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        return batch.to(self.device)
