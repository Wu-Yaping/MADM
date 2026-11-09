import os
import torch


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
