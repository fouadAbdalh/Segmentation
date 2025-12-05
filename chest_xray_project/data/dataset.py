import numpy as np
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        mask = self.masks[idx] / 255.0
        mask = np.expand_dims(mask, axis=0)

        return torch.FloatTensor(image), torch.FloatTensor(mask)