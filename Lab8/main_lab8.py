import torch

from Lab8.dictionary import findFiles, unicodeToAscii, Dictionary
from Lab8.model import RNN
from Lab8.trainer import randomTrainingExample


def load_data():
    print("load_data")


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def categoryFromOutput(output, all_categories):
    top_n, top_i = output.topk(1)  # Tensor out of Variable with .data
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


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

    print(f'[input_name[0]] = {input_name[0]}\n[hidden] = {hidden}\n[output] = {output}\n[output shape] = {output.shape}')

    category_from_output = categoryFromOutput(output, dictionary.all_categories)
    print(f'[category from output] = {category_from_output}')

    for i in range(1):
        category, line, category_tensor, line_tensor = randomTrainingExample(dictionary, device)
        print(f'category = {category} / line = {line}')
