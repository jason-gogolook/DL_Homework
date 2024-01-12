import torch
import torch.nn as nn


# 定義線性回歸模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

        # self.linear = nn.Sequential(
        #     nn.Linear(input_dim, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 50),
        #     nn.ReLU(),
        #     nn.Linear(50, 25),
        #     nn.ReLU(),
        #     nn.Linear(25, output_dim)
        # )

        # super().__init__()
        # self.layers = nn.Sequential(
        #     nn.Linear(input_dim, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 50),
        #     nn.ReLU(),
        #     nn.Linear(50, 25),
        #     nn.ReLU(),
        #     nn.Linear(25, output_dim),
        # )

    def forward(self, x):
        return self.linear(x)
        # return self.layers(x)


class linearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out
