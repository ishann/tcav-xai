#!/usr/bin/env python
# coding: utf-8
## Import packages.
import torch
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

from train import NeuralNet, preprocess_data, COMPASDataset

## Device configuration
device = "cpu"

## Hyper-parameters
INP_SIZE = 31
HID_SIZE = 8
CLASSES = 2
EPOCHS = 100
BATCH = 100
LR = 0.001

filepath="./data/compas/compas-scores-two-years.csv"
model_weights = "./data/compas_model.ckpt"


X, Y, demo = preprocess_data(filepath)

X_val, Y_val, demo_val = X[5000:,:], Y[5000:], demo[5000:,:]
data_val = COMPASDataset(X_val, Y_val)
load_val = DataLoader(data_val, batch_size=BATCH, shuffle=True)

model = NeuralNet(INP_SIZE, HID_SIZE, CLASSES).to(device)
model.load_state_dict(torch.load(model_weights))
model.eval()


gts, preds = [], []

## Evaluate the model.
with torch.no_grad():
    correct = 0
    total = 0
    for x, y in load_val:
        x = x.to(device)
        y = y.to(device)
        pred_y = model(x.float())
        _, pred_y = torch.max(pred_y.data, 1)
        total += y.size(0)
        correct += (pred_y == y).sum().item()

        gts.extend(y.tolist())
        preds.extend(pred_y.tolist())

print('Validation accuracy: {:.2f} %'.format(100 * correct / total))


# #### Assemble a dataframe to analyze.
demo_names = ["age_0", "age_1", "age_2", "age_3", "age_4", "age_5",
         "race_African-American", "race_Asian", "race_Caucasian", "race_Hispanic", "race_Native American", "race_Other",
         "sex_Female", "sex_Male"]
col_names = ["pred", "gt"] + demo_names

#print(col_names)
print(demo_val.shape)

preds = np.expand_dims(np.array(preds), axis=1)
gts = np.expand_dims(np.array(gts), axis=1)
print(preds.shape, gts.shape)

modeling_bits = np.concatenate((preds, gts), axis=1)
print(modeling_bits.shape)

data = np.concatenate((modeling_bits, demo_val), axis=1)
print(data.shape)
dd = pd.DataFrame(data=data, columns=col_names, dtype=np.int32)
dd["success"] = (dd["pred"]==dd["gt"]).astype("int")


for demo_name in demo_names:
    print("\nFrequency of {}:".format(demo_name))
    #print(dd[demo_name].value_counts(normalize=True))
    print(dd[demo_name].value_counts(normalize=False))


