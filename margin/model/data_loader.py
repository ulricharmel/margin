import random
import os
import glob

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from margin import utils 

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = transforms.Compose([
    # transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    transforms.ToTensor()])  # transform it into a torch tensor

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToTensor()])  # transform it into a torch tensor


class MSDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """

    def __init__(self, outdir, transform, split):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            outdir (string): output directory
            transform (data: transforms)
            split (str): train, val or test
        """

        self.data_dir = outdir+"/"+split
        self.imagenames = glob.glob(self.data_dir+"/*jpg")
        self.labels = [1 if "poor" in imagename else 0 for imagename in self.imagenames]

        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.imagenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        imagepath = self.imagenames[idx]
        image = Image.open(imagepath).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)


        return image, label


def fetch_dataloader(params, outdir, splits=['train', 'val', 'test']):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        GD (dict) containing all arguments for MSDataset
        params (Params) hyperparameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in splits:

        # use the train_transformer if training data, else use eval_transformer without random flip
        if split == 'train':
            dl = DataLoader(MSDataset(outdir, train_transformer, split),
                                batch_size=params.batch_size, shuffle=True,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)
        else:
            shuffle = True if split=="val" else False
            dl = DataLoader(MSDataset(outdir, eval_transformer, split),
                            batch_size=params.batch_size, shuffle=shuffle,
                            num_workers=params.num_workers,
                            pin_memory=params.cuda)

        dataloaders[split] = dl

    return dataloaders
