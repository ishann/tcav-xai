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
    print("tcav.interpret takes: {:.2f}s.".format(time.time()-start_tcav_interpret))

    return (results_feats_zeros, results_demos_zeros,
            results_feats_ones, results_demos_ones)


def assemble_scores(scores, experimental_sets, hpath, score_layer, score_type):
    """
    TODO: This is my get_pval helper functon from ./code/imagenet.
          Simplify for COMPAS.

    Will return P, N by putting positives in P and negatives in N.
    Per experimental_set, there will be
    To make ttest work, P will have repeated entries since each level has multiple -ve, but only one +ve.
    """

    P, N, idxs, names = [], [], [], []

    for concepts in experimental_sets:

        c_idxs = [c.id for c in concepts]
        c_names = [c.name for c in concepts]
        c_scores = [d.item() for d in scores["-".join([str(c.id) for c in concepts])][score_layer][score_type]]

        c_P, c_N = [], []
        for idx, c_name in enumerate(c_names):
            if c_name in hpath:
                c_P.append(c_scores[idx])
            else:
                c_N.append(c_scores[idx])

        # Hacky way to make sure equal positives and negatives
        c_P = [c_P[0] for _ in range(len(c_N))]

        P.append(c_P)
        N.append(c_N)
        idxs.append(c_idxs)
        names.append(c_names)

    return P, N, idxs, names


def get_pval(scores, experimental_sets, hpath, score_layer, score_type):
    """
    TODO: This is my pval code from ./code/imagenet.
          Simplify for COMPAS.
    """
    P, N, _, names = assemble_scores(scores, experimental_sets,
                                     score_layer, score_type)

    per_level_pvals = {}

    for set_p, set_n, set_names in zip(P, N, names):
        level = set(set_names).intersection(hpath).pop()
        # print(level, set_p, set_n)
        per_level_pvals[level] = ttest_ind(set_p, set_n)[1]

    return per_level_pvals




def short_form_explanation():
    """
    TODO: This is Kenneth's XAI from ./code/imagenet.
          Adapt to COMPAS.
    """

    per_level_pvals = get_pval(scores, experimental_sets, child_path,
                               score_layer=self.layer, score_type="magnitude")

    for level, concepts_set in zip(levels, experimental_sets):
        score_list = scores["-".join([str(c.id) for c
                            in concepts_set])][self.layer]['magnitude']
        level_explain = dict()
        level_explain['level_name'] = level
        level_explain['children'] = [(concept.name, score) for score, concept
                                     in zip(score_list, concepts_set)]

        if level in per_level_pvals:
            level_explain['pval'] = per_level_pvals[level]

        explanations.append(level_explain)

    return explanations

def long_form_explanations(self, explanations, pred_class_name):
    """
    TODO: This is Kenneth's XAI from ./code/imagenet.
          Adapt to COMPAS.
    """
    outStr = []

    prevClass = pred_class_name

    outStr.append(f"The input is predicted to be a(n) {prevClass} (p-value: {explanations[-1]['pval']:.4f}).\n")
    for explanation in explanations[::-1]:
        outStr.append(f"It is predicted to be a(n) {prevClass} because it is a {explanation['level_name']}" + ("" if 'pval' not in explanation else f" (p-value: {explanation['pval']:.4f})") + ".\n"
                      f"It is a(n) {prevClass} because out of all {explanation['level_name']}s, {prevClass} has the highest TCAV scores among possible sub-classes: \n")
        outStr.append("Class Name \t\t Score\n")
        for concept_name, concept_score in explanation['children']:
            outStr.append(f"{concept_name} \t\t {concept_score}\n")
        outStr.append("\n")

        prevClass = explanation['level_name']

    return "".join(outStr)


