import torch

from Lab8.dictionary import findFiles, unicodeToAscii, Dictionary


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
