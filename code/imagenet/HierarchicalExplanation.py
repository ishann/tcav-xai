# import ipdb

# ..........torch imports............
from pathlib import Path

import numpy as np
import torch
import torchvision
# .... Captum imports..................
from captum.attr import LayerIntegratedGradients
from captum.concept import TCAV
from captum.concept._utils.common import concepts_to_str

# .... Local imports..................
from joblib import load, dump
from torch import nn

from generate_data.hierarchy import Hierarchy
from utils import assemble_all_concepts_from_hierarchy, assemble_random_concepts, generate_experiments
from utils import load_image_tensors, transform, plot_tcav_scores, assemble_scores, get_pval, show_boxplots

# Load Hierarchy
HIERARCHY_JSON_PATH = 'generate_data/hierarchy.json'
HIERARCHY_WORDNET_PATH = 'generate_data/wordnet_labels.txt'
IMAGENET_IDX_TO_LABELS = 'generate_data/imagenet1000_clsidx_to_labels.txt'
h = Hierarchy(json_path=HIERARCHY_JSON_PATH, wordnet_labels_path=HIERARCHY_WORDNET_PATH,
              imagenet_idx_labels_path=IMAGENET_IDX_TO_LABELS)

class HierarchicalExplanation:
    def __init__(self, h: Hierarchy, model: nn.Module, layer: str, n_steps=5, load_save=False, latex_output=False):
        self.h = h
        self.model = model
        self.layer = layer
        self.n_steps = n_steps
        self.load_save = load_save
        self.tcav = TCAV(model=model, layers=[layer], layer_attr_method=LayerIntegratedGradients(self.model, None, multiply_by_inputs=False))
        self.latex_output = latex_output


    def explain(self, input_tensors, input_idx, input_class_name, get_concepts_from_name):
        assert input_class_name in h.get_leaf_nodes(), "input_class_name must be a leaf class in the hierarchy"
        levels = h.get_path(input_class_name)[:-1]
        child_path = h.get_path(input_class_name)[1:]
        explanations = []  # [{'level_name': 'name', 'children': [('name', 'value')]}]

        experimental_sets = []

        for idx, level in enumerate(levels):
            children = h.get_children_names(level)
            experiments = [get_concepts_from_name(c) for c in children]
            experiments.append(get_concepts_from_name(f'random_{idx}'))
            experimental_sets.append(experiments)

        save_name = f"{input_class_name}_scores.pkl"
        if not Path(save_name).is_file() or not self.load_save:
            scores = self.tcav.interpret(inputs=input_tensors, experimental_sets=experimental_sets, target=input_idx, n_steps=self.n_steps)
            dump(dict(scores), save_name)
        else:
            scores = load(save_name)

        per_level_pvals = get_pval(scores, experimental_sets, child_path, score_layer=self.layer, score_type="magnitude")

        for level, concepts_set in zip(levels, experimental_sets):
            score_list = scores["-".join([str(c.id) for c in concepts_set])][self.layer]['magnitude']
            level_explain = dict()
            level_explain['level_name'] = level
            level_explain['children'] = [(concept.name, score) for score, concept in zip(score_list, concepts_set)]

            pval_key = list(set([concept.name for concept in concepts_set]).intersection(per_level_pvals.keys()))[0]
            level_explain['pval'] = per_level_pvals[pval_key]

            explanations.append(level_explain)

        return explanations

    def long_form_explanations(self, explanations, pred_class_name):
        outStr = []

        prevClass = pred_class_name

        outStr.append(f"The input is predicted to be a(n) {prevClass} (p-value: {explanations[-1]['pval']:.4f}).\n")
        for explanation in explanations[::-1]:
            outStr.append(f"It is a(n) {prevClass} because out of all {explanation['level_name']}s, {prevClass} has the highest score among sub-classes: " + ("" if 'pval' not in explanation else f" (p-value: {explanation['pval']:.4f})") + "\n")

            if self.latex_output:
                outStr.append("""
                \\begin{table}[H]
                \\begin{tabular}{l|r}
                \\toprule
                \\textbf{Concept Name} & \\textbf{CAV Score}\\\\
                \\midrule
                """)
                for concept_name, concept_score in explanation['children']:
                    outStr.append(f"{concept_name} & {concept_score:.4f}\\\\ \n")

                outStr.append("""
                \\bottomrule 
                \\end{tabular}
                \\end{table}
                """)
            else:
                outStr.append("Class Name \t\t Score\n")
                for concept_name, concept_score in explanation['children']:
                    outStr.append(f"{concept_name} \t\t {concept_score:.4f}\n")
            outStr.append("\n")

            prevClass = explanation['level_name']

        return "".join(outStr)
