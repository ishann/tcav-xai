import os
import numpy as np
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


def short_form_binary_explanation(results_feats_zeros, results_demos_zeros,
                           results_feats_ones, results_demos_ones,
                           expt_set_demos, expt_set_feats,
                           score_layer, score_metric):
    """
    This is the engine. It takes in all results and expt_sets and
    generates per-feature and per-demo explanations.
    """

    feat_names = ["c_charge", "score_text", "decile_score", "priors_count"]
    demo_names = ["age", "race", "sex"]
    feat_data = {f: {} for f in feat_names}
    demo_data = {d: {} for d in demo_names}

    # Set the experiment sets for feats and demos.
    feat_data["c_charge"    ]["expt_set"] = expt_set_feats[ :2]
    feat_data["score_text"  ]["expt_set"] = expt_set_feats[ 2:5]
    feat_data["decile_score"]["expt_set"] = expt_set_feats[ 5:10]
    feat_data["priors_count"]["expt_set"] = expt_set_feats[10:]
    demo_data["age"]["expt_set"] = expt_set_demos[:6]
    demo_data["race"]["expt_set"] = expt_set_demos[6:12]
    demo_data["sex"]["expt_set"] = expt_set_demos[12:]

    # Set the results for feats.
    tmp = ['1000-1100', '1001-1101']
    feat_data["c_charge"]["results_zeros"] = [dict(results_feats_zeros[s]) for s in tmp]
    feat_data["c_charge"]["results_ones"] = [dict(results_feats_ones[s]) for s in tmp]

    tmp = ['1002-1102', '1003-1103', '1004-1104']
    feat_data["score_text"]["results_zeros"] = [dict(results_feats_zeros[s]) for s in tmp]
    feat_data["score_text"]["results_ones"] = [dict(results_feats_ones[s]) for s in tmp]

    tmp = ['1005-1105', '1006-1106', '1007-1107', '1008-1108', '1009-1109']
    feat_data["decile_score"]["results_zeros"] = [dict(results_feats_zeros[s]) for s in tmp]
    feat_data["decile_score"]["results_ones"] = [dict(results_feats_ones[s]) for s in tmp]

    tmp = ['1010-1110', '1011-1111', '1012-1112', '1013-1113', '1014-1114']
    feat_data["priors_count"]["results_zeros"] = [dict(results_feats_zeros[s]) for s in tmp]
    feat_data["priors_count"]["results_ones"] = [dict(results_feats_ones[s]) for s in tmp]

    # Set the results for demos.
    tmp = ['0-100', '1-101', '2-102', '3-103', '4-104', '5-105']
    demo_data["age"]["results_zeros"] = [dict(results_demos_zeros[s]) for s in tmp]
    demo_data["age"]["results_ones"] = [dict(results_demos_ones[s]) for s in tmp]

    tmp = ['6-106', '7-107', '8-108', '9-109', '10-110', '11-111']
    demo_data["race"]["results_zeros"] = [dict(results_demos_zeros[s]) for s in tmp]
    demo_data["race"]["results_ones"] = [dict(results_demos_ones[s]) for s in tmp]

    tmp = ['12-112', '13-113']
    demo_data["sex"]["results_zeros"] = [dict(results_demos_zeros[s]) for s in tmp]
    demo_data["sex"]["results_ones"] = [dict(results_demos_ones[s]) for s in tmp]

    # Check that lengths match up.
    for k, v in feat_data.items():
        assert len(v["expt_set"])==len(v["results_zeros"])==len(v["results_ones"])

    for k, v in demo_data.items():
        assert len(v["expt_set"])==len(v["results_zeros"])==len(v["results_ones"])


    for feat_name in feat_names:
        feat_data[feat_name]["P_zeros"] = [torch.max(el[score_layer][score_metric]).item()
                                           for el in feat_data[feat_name]["results_zeros"]]
        feat_data[feat_name]["N_zeros"] = [torch.min(el[score_layer][score_metric]).item()
                                           for el in feat_data[feat_name]["results_zeros"]]
        feat_data[feat_name]["P_ones"] = [torch.max(el[score_layer][score_metric]).item()
                                           for el in feat_data[feat_name]["results_ones"]]
        feat_data[feat_name]["N_ones"] = [torch.min(el[score_layer][score_metric]).item()
                                           for el in feat_data[feat_name]["results_ones"]]

    for demo_name in demo_names:
        demo_data[demo_name]["P_zeros"] = [el[score_layer][score_metric][0].item()
                                           for el in demo_data[demo_name]["results_zeros"]]
        demo_data[demo_name]["N_zeros"] = [el[score_layer][score_metric][1].item()
                                           for el in demo_data[demo_name]["results_zeros"]]
        demo_data[demo_name]["P_ones"] = [el[score_layer][score_metric][0].item()
                                           for el in demo_data[demo_name]["results_ones"]]
        demo_data[demo_name]["N_ones"] = [el[score_layer][score_metric][1].item()
                                           for el in demo_data[demo_name]["results_ones"]]


    for feat_name in feat_names:
         zeros_pval = ttest_ind(feat_data[feat_name]["P_zeros"], feat_data[feat_name]["N_zeros"])[-1]
         ones_pval = ttest_ind(feat_data[feat_name]["P_ones"], feat_data[feat_name]["N_ones"])[-1]
         print("{} = {:.4f}/ {:.4f}".format(feat_name, zeros_pval, ones_pval))

    for demo_name in demo_names:
         zeros_pval = ttest_ind(demo_data[demo_name]["P_zeros"], demo_data[demo_name]["N_zeros"])[-1]
         ones_pval = ttest_ind(demo_data[demo_name]["P_ones"], demo_data[demo_name]["N_ones"])[-1]
         print("{} = {:.4f}/ {:.4f}".format(demo_name, zeros_pval, ones_pval))




