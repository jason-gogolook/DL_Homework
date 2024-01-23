import glob
import os
import string
import unicodedata

import torch


def findFiles(path):
    return glob.glob(path)


def unicodeToAscii(s, all_letters=string.ascii_letters + " .,;'"):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


class Dictionary:

    def __init__(self, file_path):
        self.file_path = file_path
        self.all_letters = string.ascii_letters + " .,;'"
        self.category_lines = {}
        self.all_categories = []

    def build_category_lines(self):
        for filename in findFiles(self.file_path):
            # extract filename as the category name
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)

            # read lines from the category, which are surnames of that category
            lines = readLines(filename)
            self.category_lines[category] = lines

        n_categories = len(self.all_categories)
        print(f'[category number = {n_categories}]')
        print(f'[5 Chinese names = {self.category_lines["Chinese"][:5]}]')

    def letterToIndex(self, letter):
        return self.all_letters.find(letter)

    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    def letterToTensor(self, letter):
        tensor = torch.zeros(1, len(self.all_letters))
        tensor[0][self.letterToIndex(letter)] = 1
        return tensor

    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    def lineToTensor(self, line):
        tensor = torch.zeros(len(line), 1, len(self.all_letters))
        for li, letter in enumerate(line):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return tensor
