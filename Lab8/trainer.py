import math
import random
import time

import torch


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample(dictionary, device):
    category = randomChoice(dictionary.all_categories)
    line = randomChoice(dictionary.category_lines[category])
    category_tensor = torch.tensor([dictionary.all_categories.index(category)], dtype=torch.long)
    line_tensor = dictionary.lineToTensor(line)
    return category, line, category_tensor.to(device), line_tensor.to(device)


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {s}s'


class Trainer:

    def __init__(self, n_epochs=1):
        self.max_epochs = n_epochs

    def fit(self, model, dictionary, device):
        self.current_loss = 0
        self.all_losses = []

        self.all_categories = dictionary.all_categories
        self.category_lines = dictionary.category_lines

        # Transfer the model to the device (GPU or CPU)
        model.to(device)
