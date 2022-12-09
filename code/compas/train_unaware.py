"""
Training script for learning an _unaware_ NeuralNet on COMPAS.
"""
################################################################################
## Import packages.
################################################################################
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nets_and_datasets import NeuralNet, COMPASDataset

from tqdm import tqdm
from joblib import load, dump

import pandas as pd
import numpy as np

import ipdb

VERBOSE=False


################################################################################
## Unaware data is simply where the demogrpahic features are masked
## with the mean feature values.
################################################################################
def preprocess_unaware_data(filepath="./data/compas/compas-scores-two-years.csv",
                            unawareness=True, verbose=False, SPLIT=5000):
  """Load and process dataset from provided data path."""

  df = pd.read_csv(filepath)

  # Filter relevant features.
  features = ["age", "c_charge_degree", "race", "score_text", "sex", "priors_count",
              "days_b_screening_arrest", "decile_score", "is_recid", "two_year_recid"]

  df = df[features]
  df = df[df.days_b_screening_arrest <= 30]
  df = df[df.days_b_screening_arrest >= -30]
  df = df[df.is_recid != -1]
  df = df[df.c_charge_degree != "O"]
  df = df[df.score_text != "N/A"]

  # Divide features into: continuous, categorical and those that are continuous,
  # but need to be converted to categorical.
  categorical_features = ["c_charge_degree", "race", "score_text", "sex"]
  continuous_to_categorical_features = ["age", "decile_score", "priors_count"]

  # Bucketize features in continuous_to_categorical_features.
  for feature in continuous_to_categorical_features:
    # print(feature)
    if feature == "priors_count":
       bins = list(np.percentile(df[feature], [0, 50, 70, 80, 90, 100]))
    else:
      bins = [0] + list(np.percentile(df[feature], [20, 40, 60, 80, 90, 100]))
    print("Feature: {} || Bins: {}".format(feature, bins))
    df[feature] = pd.cut(df[feature], bins, labels=False)

  # Binarize all categorical features (including the ones bucketized above).
  df = pd.get_dummies(df, columns=categorical_features +
                      continuous_to_categorical_features)

  # Define protected attributes after one-hot-encoding.
  prot_attributes = ['age_0', 'age_1', 'age_2', 'age_3', 'age_4', 'age_5',
                     'race_African-American', 'race_Asian', 'race_Caucasian',
                     'race_Hispanic', 'race_Native American', 'race_Other',
                     'sex_Female', 'sex_Male']

  # Fill values for decile scores and prior counts feature buckets.
  to_fill = [u"decile_score_0", u"decile_score_1", u"decile_score_2",
             u"decile_score_3", u"decile_score_4", u"decile_score_5"]

  for i in range(len(to_fill) - 1):
    df[to_fill[i]] = df[to_fill[i:]].max(axis=1)
  to_fill = [u"priors_count_0.0", u"priors_count_1.0", u"priors_count_2.0",
             u"priors_count_3.0", u"priors_count_4.0"]
  for i in range(len(to_fill) - 1):
    df[to_fill[i]] = df[to_fill[i:]].max(axis=1)

  # Get the labels (two year recidivism) and groups (female defendants).
  labels = df["two_year_recid"]
  groups = df[prot_attributes]

  # Retain all features other than "two_year_recid" and "is_recid".
  df.drop(columns=["two_year_recid", "is_recid"], inplace=True)

  if unawareness == True:
      for col in df.columns:
          if "age" in col or "race" in col or "sex" in col:
              df[col][:SPLIT] = df[col][:SPLIT].mean()

  #ipdb.set_trace()

  if verbose==True:
      print("Columns in data:")
      for idx, col in enumerate(df.columns.tolist()):
          print(idx, col)

      print("Columns in demo:")
      for idx, col in enumerate(groups.columns.tolist()):
          print(idx, col)


  #cols_to_drop = ["days_b_screening_arrest", "score_text_High",
  #                "score_text_Low", "score_text_Medium",
  #                "decile_score_0", "decile_score_1",
  #                "decile_score_2", "decile_score_3",
  #                "decile_score_4", "decile_score_5"]

  #df.drop(cols_to_drop, axis=1, inplace=True)

  return df.to_numpy(), labels.to_numpy(), groups.to_numpy()


################################################################################
## Device configuration
################################################################################
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"


################################################################################
## Hyper-parameters
################################################################################
INP_SIZE = 31
HID_SIZE = 8
CLASSES = 2
EPOCHS = 100
BATCH = 100
LR = 0.001


################################################################################
## Dataset and dataloader.
################################################################################
filepath="./data/compas/compas-scores-two-years.csv"
SPLIT = 5000
X, Y, demo = preprocess_unaware_data(filepath, verbose=True, SPLIT=5000)

# Manually split data to keep track of demo as well.
X_trn, Y_trn, demo_trn = X[:SPLIT,:], Y[:SPLIT], demo[:SPLIT,:]
X_val, Y_val, demo_val = X[SPLIT:,:], Y[SPLIT:], demo[SPLIT:,:]

data_trn = COMPASDataset(X_trn, Y_trn)
data_val = COMPASDataset(X_val, Y_val)

load_trn = DataLoader(data_trn, batch_size=BATCH, shuffle=True)
load_val = DataLoader(data_val, batch_size=BATCH, shuffle=True)


################################################################################
## Loss and optimizer
################################################################################
model = NeuralNet(INP_SIZE, HID_SIZE, CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


################################################################################
## Training loop.
################################################################################
total_steps = len(load_trn)
print("Training...")
for epoch in tqdm(range(EPOCHS)):
    for i, (x, y) in enumerate(load_trn):
        # Move tensors to the configured device
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        pred_y = model(x.float())
        loss = criterion(pred_y, y)

        # Backward pass and optimize.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if VERBOSE:
        print('[{}/{}] = Loss: {:.4f}'.format(epoch+1, EPOCHS, loss.item()))

################################################################################
## Test the model.
################################################################################
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

print('Validation accuracy: {:.2f} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), './data/compas_unaware_model.ckpt')

# Save actual data.
data = {"X": X_val,
        "Y": Y_val,
        "demo": demo_val}
dump(data, "./data/val_unaware_data.pkl")


