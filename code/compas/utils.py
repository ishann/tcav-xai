"""
Methods for doing practically everything in main.py.
"""
import os
import numpy as np
import time
import ipdb

import torch

from captum.concept import Concept
from captum.concept._utils.common import concepts_to_str
from captum.concept._utils.data_iterator import dataset_to_dataloader
from captum.attr import LayerIntegratedGradients
from captum.concept import TCAV

from nets_and_datasets import COMPAS_Concept_Dataset
from scipy.stats import ttest_ind

from joblib import load, dump

#import warnings
#warnings.resetwarnings()
#warnings.simplefilter('ignore', UserWarning)

#from torchvision import transforms
#from tqdm import tqdm

def assemble_data_tensors(data_path):

    dd = load(data_path)
    X, Y = dd["X"], dd["Y"]

    #zeros_mask, ones_mask = (Y==0), (Y==1)
    zeros_tensor = torch.from_numpy(X[Y==1]).float()
    ones_tensor = torch.from_numpy(X[Y==0]).float()

    zeros_idx, ones_idx = 0, 1

    return zeros_tensor, ones_tensor, zeros_idx, ones_idx


def assemble_concepts_and_expt_sets(concepts_table_demos_path,
                                    concepts_table_feats_path):

    concepts_tables_demos = load(concepts_table_demos_path)
    concepts_tables_feats = load(concepts_table_feats_path)

    expt_set_demos = []
    for idx, (key, (intr, disr)) in enumerate(concepts_tables_demos.items()):

        #print(idx, key, intr.shape, disr.shape)

        intr = intr.astype(np.float32)
        disr = disr.astype(np.float32)

        concept_interest = assemble_concept(name="{}_interest".format(key),
                                            id=0+idx, concept_tensor=intr)
        concept_disinterest = assemble_concept(name="{}_disinterest".format(key),
                                              id=100+idx, concept_tensor=disr)

        expt_set_demos.append([concept_interest, concept_disinterest])

    expt_set_feats = []
    for idx, (key, (intr, disr)) in enumerate(concepts_tables_feats.items()):

        #print(idx, key, intr.shape, disr.shape)

        intr = torch.tensor(intr).float()
        disr = torch.tensor(disr).float()

        concept_interest = assemble_concept(name="{}_interest".format(key),
                                            id=1000+idx, concept_tensor=intr)
        concept_disinterest = assemble_concept(name="{}_disinterest".format(key),
                                              id=1100+idx, concept_tensor=disr)

        expt_set_feats.append([concept_interest, concept_disinterest])


    return expt_set_demos, expt_set_feats


def assemble_concept(name, id, concept_tensor):
    """
    The original Concept uses an iterable form of dataloaders.
    But the documentation doesn't specify that it must be an IterableDataset.
    For now, our COMPAS_Concept_Dataset is inherited from a vanilla Dataset.
    This might be a pain later on, and a source for silent errors.
    """
    #concept_path = os.path.join(concepts_path, name) + "/"

    dataset = COMPAS_Concept_Dataset(concept_tensor)
    concept_loader = dataset_to_dataloader(dataset)

    return Concept(id=id, name=name, data_iter=concept_loader)


def compute_tcavs(model, layers, zeros_tensor, ones_tensor,
                  zeros_idx, ones_idx, expt_set_feats, expt_set_demos):

    start_tcav_interpret = time.time()

    # Create TCAV for each experiment
    layer_attr_method = LayerIntegratedGradients(model, None,
                                                 multiply_by_inputs=False)
    model_tcav_zeros = TCAV(model=model, layers=layers,
                            layer_attr_method=layer_attr_method)

    start_tcav_interpret = time.time()
    results_feats_zeros = model_tcav_zeros.interpret(inputs=zeros_tensor,
                                                experimental_sets=expt_set_feats,
                                                target=zeros_idx, n_steps=5)
    results_demos_zeros = model_tcav_zeros.interpret(inputs=zeros_tensor,
                                                experimental_sets=expt_set_demos,
                                                target=zeros_idx, n_steps=5)

    model_tcav_ones = TCAV(model=model, layers=layers,
                           layer_attr_method=layer_attr_method)
    results_feats_ones = model_tcav_ones.interpret(inputs=ones_tensor,
                                                experimental_sets=expt_set_feats,
                                                target=ones_idx, n_steps=5)
    results_demos_ones = model_tcav_ones.interpret(inputs=ones_tensor,
                                                experimental_sets=expt_set_demos,
                                                target=ones_idx, n_steps=5)

    print("tcav.interpret takes: {:.2f}s.".format(time.time()-
                                                  start_tcav_interpret))

    results_feats_zeros = dict(results_feats_zeros)
    results_demos_zeros = dict(results_demos_zeros)
    results_feats_ones = dict(results_feats_ones)
    results_demos_ones = dict(results_demos_ones)

    return (results_feats_zeros, results_demos_zeros,
            results_feats_ones, results_demos_ones)



