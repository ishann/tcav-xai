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

from HierarchicalExplanation import HierarchicalExplanation
from generate_data.hierarchy import Hierarchy
from utils import assemble_all_concepts_from_hierarchy, assemble_random_concepts, generate_experiments
from utils import load_image_tensors, transform, plot_tcav_scores, assemble_scores, get_pval, show_boxplots

# Load Hierarchy
HIERARCHY_JSON_PATH = 'generate_data/hierarchy.json'
HIERARCHY_WORDNET_PATH = 'generate_data/wordnet_labels.txt'
IMAGENET_IDX_TO_LABELS = 'generate_data/imagenet1000_clsidx_to_labels.txt'
h = Hierarchy(json_path=HIERARCHY_JSON_PATH, wordnet_labels_path=HIERARCHY_WORDNET_PATH,
              imagenet_idx_labels_path=IMAGENET_IDX_TO_LABELS)

###################################################
# Assemble Concepts
# Let's assemble concepts into Concept instances using Concept class and concept images stored in `concepts_path`.
###################################################

# concepts_path = "/home/devvrit/ishann/data/captum/tcav/concepts"
concepts_path = "../data"

# Assemble non-random concepts
concepts = assemble_all_concepts_from_hierarchy(h=h, num_images=100, concepts_path=concepts_path,
                                                recreate_if_exists=True)  # Only 100 images for testing, can increase later

# Assemble all random concepts
random_concepts = assemble_random_concepts(concepts_path=concepts_path)

##############################
# Experiments
##############################

# Defining GoogleNet Model
model = torchvision.models.googlenet(pretrained=True)
model = model.eval()
# model.inception5b is the last inception module.
# model.fc is the fully-connected layer outputting a 1000-way embedding.
# layers = ['inception5b', 'fc']
layers=['fc']

# Generate experiment sets
# experiments = generate_experiments(concepts_dict=concepts, random_concepts_dict=random_concepts)
# print(experiments)

experiment_w_random = [[concepts['canine'], random_concepts[0]], [concepts['canine'], random_concepts[1]]]
experiment_w_comparison = [[concepts['canine'], concepts['tabby'], concepts['puzzle'], concepts['valley']]]

# Create TCAV for each experiment
tcav = TCAV(model=model, layers=layers, layer_attr_method=LayerIntegratedGradients(model, None, multiply_by_inputs=False))

# Load sample images
husky_images = load_image_tensors('Siberian husky', root_path=concepts_path, transform=False, count=100)
husky_tensors = torch.stack([transform(img) for img in husky_images])
husky_idx = h.imagenet_label2idx['Siberian husky']

# valley_images = load_image_tensors('valley', root_path=concepts_path, transform=False, count=100)
# valley_idx = h.imagenet_label2idx['valley']
#
# tabby_images = load_image_tensors('tabby', root_path=concepts_path, transform=False, count=100)
# tabby_idx = h.imagenet_label2idx['tabby']

# print("Starting...")
# if not Path("husky_tcav_w_random.pkl").is_file():
#     husky_tcav_w_random = tcav.interpret(inputs=husky_tensors, experimental_sets=experiment_w_random, target=husky_idx, n_steps=5)
#     print(husky_tcav_w_random)
#     # dump(husky_tcav_w_random, "husky_tcav_w_random.pkl")
# else:
#     pass
#     # husky_tcav_w_random = load("husky_tcav_w_random.pkl")

# plot_tcav_scores(layers=layers, experimental_sets=experiment_w_random, tcav_scores=husky_tcav_w_random, outfile="husky_vs_rand.png")
# plot_tcav_scores(layers=layers, experimental_sets=experiment_w_random, tcav_scores=husky_tcav_w_comparison, outfile="husky_vs_others.png"

# # p-val
# experimental_sets = [[concepts['canine'], random_concepts[0]], [concepts['canine'], random_concepts[1]],
#                      [random_concepts[0], random_concepts[1]], [random_concepts[0], random_concepts[2]]]
#
# print("starting pval")
# if not Path("scores.pkl").is_file():
#     scores = tcav.interpret(inputs=husky_tensors, experimental_sets=experimental_sets, target=husky_idx, n_steps=5)
#     # dump(scores, "scores.pkl")
# else:
#     pass
#     # scores = load("scores.pkl")
# print("scores")
# print(scores)
#
# print("boxplots")
# # show_boxplots(scores=scores, experimental_sets=experimental_sets, n=4, layer='inception5b', outfile="boxplot_inception5b.png")
# show_boxplots(scores=scores, experimental_sets=experimental_sets, n=2, layer='fc', outfile="boxplot_fc.png")


he = HierarchicalExplanation(h=h, model=model, layer='fc', n_steps=5, load_save=True)
explanations = he.explain(input_tensors=husky_tensors, input_class_name="Siberian husky", input_idx=husky_idx, get_concepts_from_name=lambda x: concepts[x] if x in concepts else random_concepts[int(x.replace("random_", ""))])
print(explanations)
long_form = he.long_form_explanations(explanations, "Siberian husky")
print(long_form)
















# Run tests to get TCAV scores
# tcav_scores_husky = tcav.interpret(inputs=husky_images, experimental_sets=experiments, target=husky_idx, n_steps=5)
# print("tcav husky")
# print(tcav_scores_husky)
#
# tcav_scores_valley = tcav.interpret(inputs=valley_images, experimental_sets=experiments, target=valley_idx, n_steps=5)
# print("tcav valley")
# print(tcav_scores_valley)
#
# tcav_scores_tabby = tcav.interpret(inputs=tabby_images, experimental_sets=experiments, target=tabby_idx, n_steps=5)
# print("tcav tabby")
# print(tcav_scores_tabby)

# multi_concept = "canine"
# # concepts = ["siberean_husky", "white_wolf"]
# concepts = ["Siberian husky", "white wolf"]
# canine_concept = assemble_multi_concept(multi_concept, concepts, 0, concepts_path=concepts_path)
#
# #stripes_concept = assemble_concept("striped", 0, concepts_path=concepts_path)
# #zigzagged_concept = assemble_concept("zigzagged", 1, concepts_path=concepts_path)
# #dotted_concept = assemble_concept("dotted", 2, concepts_path=concepts_path)
#
# random_0_concept = assemble_concept("random_0", 1, concepts_path=concepts_path)
# random_1_concept = assemble_concept("random_1", 2, concepts_path=concepts_path)
#
# # ## Defining GoogleNet Model
#
# # For this tutorial, we will load GoogleNet model and set in eval mode.
# model = torchvision.models.googlenet(pretrained=True)
# model = model.eval()
#
# # # Computing TCAV Scores
#
# # Next, let's create TCAV class by passing the instance of GoogleNet model, a custom classifier and the list of layers where we would like to test the importance of concepts.
#
# # The custom classifier will be trained to learn classification boundaries between concepts. We offer a default implementation of Custom Classifier in captum library so that the users do not need to define it. Captum users, hoowever, are welcome to define their own classifers with any custom logic. CustomClassifier class extends abstract Classifier class and provides implementations for training workflow for the classifier and means to access trained weights and classes. Typically, this can be, but is not limited to a classier, from sklearn library. In this case CustomClassifier wraps `linear_model.SGDClassifier` algorithm from sklearn library.
#
# # model.inception5b is the last inception module.
# # model.fc is the fully-connected layer outputting a 1000-way embedding.
# layers=['inception5b', 'fc']
#
# #ipdb.set_trace()
# mytcav = TCAV(model=model,
#               layers=layers,
#               layer_attr_method = LayerIntegratedGradients(
#                 model, None, multiply_by_inputs=False))
#
#
# # ##### Defining experimantal sets for CAV
#
# # Then, lets create 2 experimental sets: ["striped", "random_0"] and ["striped", "random_1"].
# #
# # We will train a classifier for each experimal set that learns hyperplanes which separate the concepts in each experimental set from one another. This is especially interesting when later we want to perform two-sided statistical significance testing in order to confirm or reject the significance of a concept in a specific layer for predictions.
# #
# # Now, let's load sample images from imagenet. The goal is to test how sensitive model predictions are to predefined concepts such as 'striped' when predicting 'zebra' class.
# experimental_set_rand = [[canine_concept, random_0_concept], [canine_concept, random_1_concept]]
#
#
# # Now, let's load sample images from imagenet. The goal is to test how sensitive model predictions are to predefined concepts such as 'striped' when predicting 'zebra' class.
# #
# # Please, download zebra images and place them under `data/tcav/image/imagenet/zebra` folder in advance before running the cell below. For our experiments we used 50 different zebra images from imagenet dataset.
#
#
# # Load sample images from folder
# # husky_imgs = load_image_tensors('siberean_husky', root_path=ROOT_PATH, transform=False)
# husky_imgs = load_image_tensors('Siberian husky', root_path=concepts_path, transform=False)
#
# # Here we perform a transformation and convert the images into tensors, so that we can use them as inputs to NN model.
#
# # Load sample images from folder
# husky_tensors = torch.stack([transform(img) for img in husky_imgs])
# print(experimental_set_rand)
#
#
# # zebra class index
# # siberean_husky = 250
# # white_wolf = 270
# canine_ind = 250
#
# tcav_scores_w_random = mytcav.interpret(inputs=husky_tensors,
#                                         experimental_sets=experimental_set_rand,
#                                         target=canine_ind,
#                                         n_steps=5,
#                                        )
#
# print(tcav_scores_w_random)
