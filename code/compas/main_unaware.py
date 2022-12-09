"""
Script for load pretrained model, concept tensors, and training TCAVs.
"""
################################################################################
##
################################################################################
import ipdb

import torch

from nets_and_datasets import NeuralNet, COMPAS_Concept_Dataset
from utils import assemble_data_tensors, assemble_concepts_and_expt_sets
from utils import compute_tcavs#, short_form_binary_explanation
from xai import short_form_binary_explanation

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

################################################################################
##  Load model.
################################################################################
INP_SIZE, HID_SIZE, CLASSES = 31, 8, 2
model_weights = "./data/compas_unaware_model.ckpt"
model = NeuralNet(INP_SIZE, HID_SIZE, CLASSES).to("cpu")
model.load_state_dict(torch.load(model_weights))
model.eval()


################################################################################
##  Assemble data tensors (similar to Siberean Husky).
################################################################################
print("Assembling data tensors...")
data_path = "./data/val_unaware_data.pkl"
(zeros_tensor,
 ones_tensor,
 zeros_idx,
 ones_idx)    = assemble_data_tensors(data_path)


################################################################################
##  Assemble concepts and experiment sets.
################################################################################
print("Assembling concepts and experiment sets...")
concepts_table_demos_path = "./data/concepts_tables_demos.pkl"
concepts_table_feats_path = "./data/concepts_tables_unaware_feats.pkl"

(expt_set_demos,
 expt_set_feats) = assemble_concepts_and_expt_sets(concepts_table_demos_path,
                                                   concepts_table_feats_path)


################################################################################
##  TCAV.
################################################################################
print("Starting TCAVing...")
layers=["fc4"]
(results_feats_zeros,
 results_demos_zeros,
 results_feats_ones,
 results_demos_ones) = compute_tcavs(model, layers,
                                     zeros_tensor, ones_tensor,
                                     zeros_idx, ones_idx,
                                     expt_set_feats, expt_set_demos)

assert len(results_feats_zeros)==len(results_feats_ones)==len(expt_set_feats), \
       ("Feature TCAV dimensions do not match experiment sets.")

assert len(results_demos_zeros)==len(results_demos_ones)==len(expt_set_demos), \
       ("Demo TCAV dimensions do not match experiment sets.")


print("Starting short form explanations...")
score_layer = "fc4"
score_metric = "magnitude"

short_form_binary_explanation(results_feats_zeros, results_demos_zeros,
                              results_feats_ones, results_demos_ones,
                              expt_set_demos, expt_set_feats,
                              score_layer, score_metric)

# ipdb.set_trace()


