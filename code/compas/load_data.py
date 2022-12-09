"""
Standalone script to create dicts of numpy arrays that contain concepts.
"""
#!/usr/bin/env python3
################################################################################
##  Import packages.
################################################################################
import glob
import os
from typing import Callable, Iterator

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np
from joblib import load, dump


################################################################################
##  Define things. Load data.
################################################################################
features = ["days_b_screening_arrest",
            "c_charge_degree_F",
            "c_charge_degree_M",
            "race_African-American",
            "race_Asian",
            "race_Caucasian",
            "race_Hispanic",
            "race_Native American",
            "race_Other",
            "score_text_High",
            "score_text_Low",
            "score_text_Medium",
            "sex_Female",
            "sex_Male",
            "age_0",
            "age_1",
            "age_2",
            "age_3",
            "age_4",
            "age_5",
            "decile_score_0",
            "decile_score_1",
            "decile_score_2",
            "decile_score_3",
            "decile_score_4",
            "decile_score_5",
            "priors_count_0.0",
            "priors_count_1.0",
            "priors_count_2.0",
            "priors_count_3.0",
            "priors_count_4.0"]

demos = [ "age_0",
           "age_1",
           "age_2",
           "age_3",
           "age_4",
           "age_5",
           "race_African-American",
           "race_Asian",
           "race_Caucasian",
           "race_Hispanic",
           "race_Native American",
           "race_Other",
           "sex_Female",
           "sex_Male"]

data_path = "./data/val_data.pkl"
dd = load(data_path)

X, Y, demo = dd["X"], np.expand_dims(dd["Y"], axis=1), dd["demo"]

table = np.concatenate([Y, X], axis=1)
columns = ["outcome"] + features

data = pd.DataFrame(data=table, columns=columns, dtype=np.int64)


################################################################################
##  Assemble concepts.
################################################################################
def assemble_concept_tables(data, features, column_of_interest,
                            column_of_disinterest=None, verbose=False):
    """
    data                 : Entire COMPAS validation data.
    features             : Features to return in concept tables.
    column_of_interest   : Feature we're interested in, say, age_0.
    column_of_disinterest: Opposite of interesting feature, say, age_5.
                           Can be set to None => a random set of data points are
                           sampled where column_of_interest==0.
    """

    interesting_concept = data[data[column_of_interest]==1][features]

    if column_of_disinterest is None:
        num_disinteresting = min(len(data)-len(interesting_concept), len(interesting_concept))
        disinteresting_concept = data[data[column_of_interest]==0][features].sample(n=num_disinteresting)
    else:
        disinteresting_concept = data[data[column_of_disinterest]==1][features]

    num_interesting = len(interesting_concept)
    num_disinteresting = len(disinteresting_concept)

    if verbose:
        print("Number of interesting concepts: {}".format(num_interesting))
        print("Number of disinteresting concepts: {}".format(num_disinteresting))

    return interesting_concept.to_numpy(), disinteresting_concept.to_numpy()

column_of_interest = "race_African-American"
column_of_disinterest = None # "age_5"

interesting, disinteresting = assemble_concept_tables(data=data,
                                features=features,
                                column_of_interest=column_of_interest,
                                column_of_disinterest=column_of_disinterest)


# Remove demographics and decile_score_0 and days_b_screening_arrest.
unaware_feats = [f for f in features if f not in demos]
unaware_feats.remove("decile_score_0")
unaware_feats.remove("days_b_screening_arrest")

concepts_unaware_feats = {}
concepts_demos = {}

for unaware_feat in unaware_feats:
    interesting, disinteresting = assemble_concept_tables(data, features, unaware_feat)
    concepts_unaware_feats[unaware_feat] = [interesting, disinteresting]

for demo in demos:
    interesting, disinteresting = assemble_concept_tables(data, features, demo)
    concepts_demos[demo] = [interesting, disinteresting]

# Sanity check befor dumping to disk.
for concept, [interesting, disinteresting] in concepts_unaware_feats.items():
    print(concept, interesting.shape, disinteresting.shape)

for concept, [interesting, disinteresting] in concepts_demos.items():
    print(concept, interesting.shape, disinteresting.shape)

dump(concepts_unaware_feats, "./data/concept_tables_unaware_feats.pkl")
dump(concepts_demos, "./data/concepts_tables_demos.pkl")




