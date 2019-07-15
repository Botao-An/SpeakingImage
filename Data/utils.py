import os
from PIL import Image
import pandas as pd
import numpy as np
import torch

def get_files(folder):

    if not os.path.isdir(folder):
        raise RuntimeError(" \"{0}\" is not a folder. " .format(folder))

    filtered_files = []

    for path, _, files in os.walk(folder):
        files.sort()
        for file in files:
            full_path = os.path.join(path, file)
            filtered_files.append(full_path)

    return filtered_files

def pil_loader(data_path):

    data = Image.open(data_path).convert('RGB')

    return data

def csv_loader(data_path):

    data = pd.read_csv(data_path, header=None, dtype='float32', encoding='unicode_escape')
    data = np.array(data)[:,:,np.newaxis]
    data = np.transpose(data, (2, 0, 1))
    data = torch.from_numpy(data)

    return data