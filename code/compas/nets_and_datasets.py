"""
Classes and methods for setting up network training.
"""
################################################################################
## Import packages.
################################################################################
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#from utils_train import NeuralNet, preprocess_data, COMPASDataset

from tqdm import tqdm
from joblib import load, dump

import ipdb

import pandas as pd
import numpy as np

import warnings
warnings.resetwarnings()
warnings.simplefilter('ignore', UserWarning)


################################################################################
## Utilities for training a base neural network on COMPAS for TCAV.
################################################################################
# Network.
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):

        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)

        return out


################################################################################
##  Class for managing COMPAS dataset.
################################################################################
class COMPASDataset(Dataset):
    """
    This Dataset is very simple because preprocess_data() did all the work.
    Ideally, preprocess_data should be a method belonging to this class.
    """
    def __init__(self, X, Y):

        self.X = X
        self.Y = Y


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return [self.X[idx], self.Y[idx]]


################################################################################
## Class for defining concepts. Both feats and demos are supported.
################################################################################
class COMPAS_Concept_Dataset(Dataset):
    """
    A custom class to iterate through concept tables.
    """
    def __init__(self, concept_tensor):
        """

        """
        self.concept_data = torch.tensor(concept_tensor).float()

    def __len__(self):

        return len(self.concept_data)

    def __getitem__(self, idx):

        return self.concept_data[idx]



