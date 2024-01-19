import glob
import string
import unicodedata

import torch


def load_data():
    print("load_data")


def find_files(path):
    return glob.glob(path)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def unicodeToAscii(s, all_letters=string.ascii_letters + " .,;'"):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def build_category_lines():
    print("build_category_lines")


if __name__ == '__main__':
    file_path = 'language_data/names/*.txt'
    print('[All files]: \n', find_files(file_path))

    device = get_default_device()
    print('[Device]: \n', device)

    print('[UnicodeToAscii] example: Ślusàrski: \n', unicodeToAscii('Ślusàrski'))
