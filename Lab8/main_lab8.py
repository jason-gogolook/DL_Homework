import torch

from Lab8.dictionary import findFiles, unicodeToAscii, Dictionary
from Lab8.model import RNN
from Lab8.trainer import randomTrainingExample, categoryFromOutput, Trainer

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_data():
    print("load_data")


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


if __name__ == '__main__':
    file_path = 'language_data/names/*.txt'
    print('[All files]: \n', findFiles(file_path))

    device = get_default_device()
    print('[Device]: \n', device)

    print('[UnicodeToAscii] example: Ślusàrski: \n', unicodeToAscii('Ślusàrski'))

    dictionary = Dictionary(file_path)
    dictionary.build_category_lines()
    print(f'[\'y\' to index = {dictionary.letterToIndex("y")}\n]')
    print(f'[\'y\' to tensor = {dictionary.letterToTensor("y")}\n]')
    print(f'[\'Jones\' to tensor = {dictionary.lineToTensor("Jones")}'
          f'\n shape = {dictionary.lineToTensor("Jones").shape}\n]')

    print(dictionary.show_info())

    # size of the hidden state
    n_hidden = 128
    input_name = dictionary.lineToTensor('Albert')
    input_name = input_name.to(device)

    # As it is the first step, we need to initialize the hidden layer

    # For RNN
    hidden = torch.zeros(1, n_hidden).to(device)
    n_letters = len(dictionary.all_letters)
    n_categories = len(dictionary.all_categories)
    model = RNN(n_letters, n_hidden, n_categories, lr=1e-04)
    model = model.to(device)

    output, next_hidden = model(input_name[0], hidden)

    print(
        f'[input_name[0]] = {input_name[0]}\n[hidden] = {hidden}\n[output] = {output}\n[output shape] = {output.shape}')

    category_from_output = categoryFromOutput(output, dictionary.all_categories)
    print(f'[category from output] = {category_from_output}')

    for i in range(1):
        category, line, category_tensor, line_tensor = randomTrainingExample(dictionary, device)
        print(f'category = {category} / line = {line}')

    # 3. Training the network
    # 3.1. Creating the trainer class - note that here, I passed writer as a  parameter to the trainer
    trainer = Trainer(n_epochs=1)
    trainer.fit(model, dictionary, device)

    # Plotting the result
    # Plotting the historical loss from ``all_losses`` shows the network learning.
    plt.figure()
    all_losses = trainer.all_losses

    # Convert tensors back to cpu because numpy is unable to use gpus
    all_losses_cpu = [loss.cpu().item() for loss in all_losses]
    plt.title('Loss vs Epochs')
    plt.plot(all_losses_cpu)

    # Evaluating the results
    #   To see how well the network performs on different categories, we will create a confusion matrix,
    #   indicating for every actual language (rows) which language the network guesses (columns).
    #   To calculate the confusion matrix a bunch of samples are run through the network with evaluate(),
    #   which is the same as fit() minus the backprop.

    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # 当你调用model.to(device) 时，PyTorch会将模型内的所有参数和缓冲区复制到device指定的设备上。
    # 如果device是GPU，模型的计算将在GPU上进行，这通常比在CPU上快得多。
    model = model.to(device)

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample(dictionary, device)
        output = model.evaluate(line_tensor, device)

        guess, guess_i = categoryFromOutput(output, dictionary.all_categories)
        category_i = dictionary.all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + dictionary.all_categories, rotation=90)
    ax.set_yticklabels([''] + dictionary.all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.title('Confusion matrix')
    plt.show()
