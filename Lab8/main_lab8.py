import torch

from Lab8.dictionary import findFiles, unicodeToAscii, Dictionary
from Lab8.model import RNN


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

    print(f'[output] = {output}')
