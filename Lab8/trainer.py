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


def categoryFromOutput(output, all_categories):
    top_n, top_i = output.topk(1)  # Tensor out of Variable with .data
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


class Trainer:

    def __init__(self, n_epochs=1):
        self.max_epochs = n_epochs

    def fit(self, model, dictionary, device):
        self.current_loss = 0
        self.all_losses = []

        # Transfer the model to the device (GPU or CPU)
        model.to(device)

        # Configure the optimizer
        self.optimizer = model.configure_optimizers()
        self.model = model

        self.start = time.time()

        for epoch in range(self.max_epochs):
            self.fit_epoch(dictionary, device)

            # Logging the average training loss so that it can be visualized in the tensorboard
            # self.writer.add_scalar("Training Loss", self.avg_training_loss, epoch)

        print("Training process has finished")

    def fit_epoch(self, dic, device):
        n_iters = 100000
        print_every = 5000
        plot_every = 1000

        self.current_loss = 0.0
        self.all_losses = []

        # iterate over the DataLoader for training data
        for iter in range(1, n_iters + 1):
            # Get input
            category, line, category_tensor, line_tensor = randomTrainingExample(dic, device)

            # training
            hidden = self.model.initHidden(device)

            # Clear gradient buffers because we don't want any gradient from previous
            # epoch to carry forward, don't want to accumulate gradients
            self.optimizer.zero_grad()

            # get output from the model, given the inputs
            for i in range(line_tensor.size()[0]):
                output, hidden = self.model(line_tensor[i], hidden)

            # get loss for the predicted output
            loss = self.model.loss(output, category_tensor)

            # get gradients w.r.t to the parameters of the model
            loss.backward()

            # update the parameters (perform optimization)
            self.optimizer.step()

            # Let's print some statistics - Gradient is not required from here
            with torch.no_grad():
                self.current_loss += loss

                # Print iter number, loss, name and guess
                if iter % print_every == 0:
                    guess, guess_i = categoryFromOutput(output, dic.all_categories)
                    correct = '✓' if guess == category else f'✗ ({category})'
                    print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(self.start), loss, line, guess, correct))

                # Add current loss avg to list of losses (average loss of "plot_every" iterations)
                if iter % plot_every == 0:
                    self.all_losses.append(self.current_loss / plot_every)
                    self.current_loss = 0
