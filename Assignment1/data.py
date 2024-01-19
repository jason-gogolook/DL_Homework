import numpy as np
import torch
from torch.utils.data import Dataset


class HouseData(Dataset):
    def __init__(self, data, label):
        # 假设csv文件的每一行是一个样本，每一列是一个特征
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample = self.data.drop("SalePrice", axis=1).iloc[idx, :].to_numpy()
        # # label = np.log1p(self.data["SalePrice"][idx])
        # label = self.data["SalePrice"][idx]

        data = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.label[idx], dtype=torch.float32)

        return data, label
