import ipdb

import numpy as np
import os, glob

import matplotlib.pyplot as plt

from PIL import Image

from scipy.stats import ttest_ind

# ..........torch imports............
import torch
import torchvision

from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms

#.... Captum imports..................
from captum.attr import LayerGradientXActivation, LayerIntegratedGradients

from captum.concept import TCAV
from captum.concept import Concept

from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.concept._utils.common import concepts_to_str

#.... Local imports..................
from utils import transform, get_tensor_from_filename,load_image_tensors
from utils import assemble_concept, assemble_multi_concept

ROOT_PATH = "/home/devvrit/ishann/data/captum/tcav/concepts"

# Let's assemble concepts into Concept instances using Concept class and concept images stored in `concepts_path`.
#ipdb.set_trace()

concepts_path = ROOT_PATH

multi_concept = "canine"
concepts = ["siberean_husky", "white_wolf"]
canine_concept = assemble_multi_concept(multi_concept, concepts, 0, concepts_path=concepts_path)

#stripes_concept = assemble_concept("striped", 0, concepts_path=concepts_path)
#zigzagged_concept = assemble_concept("zigzagged", 1, concepts_path=concepts_path)
#dotted_concept = assemble_concept("dotted", 2, concepts_path=concepts_path)

random_0_concept = assemble_concept("random_0", 1, concepts_path=concepts_path)
random_1_concept = assemble_concept("random_1", 2, concepts_path=concepts_path)

# ## Defining GoogleNet Model

# For this tutorial, we will load GoogleNet model and set in eval mode.
model = torchvision.models.googlenet(pretrained=True)
model = model.eval()

# # Computing TCAV Scores

# Next, let's create TCAV class by passing the instance of GoogleNet model, a custom classifier and the list of layers where we would like to test the importance of concepts.

# The custom classifier will be trained to learn classification boundaries between concepts. We offer a default implementation of Custom Classifier in captum library so that the users do not need to define it. Captum users, hoowever, are welcome to define their own classifers with any custom logic. CustomClassifier class extends abstract Classifier class and provides implementations for training workflow for the classifier and means to access trained weights and classes. Typically, this can be, but is not limited to a classier, from sklearn library. In this case CustomClassifier wraps `linear_model.SGDClassifier` algorithm from sklearn library.

# model.inception5b is the last inception module.
# model.fc is the fully-connected layer outputting a 1000-way embedding.
layers=['inception5b', 'fc']

#ipdb.set_trace()
mytcav = TCAV(model=model,
              layers=layers,
              layer_attr_method = LayerIntegratedGradients(
                model, None, multiply_by_inputs=False))


# ##### Defining experimantal sets for CAV

# Then, lets create 2 experimental sets: ["striped", "random_0"] and ["striped", "random_1"].
#
# We will train a classifier for each experimal set that learns hyperplanes which separate the concepts in each experimental set from one another. This is especially interesting when later we want to perform two-sided statistical significance testing in order to confirm or reject the significance of a concept in a specific layer for predictions.
#
# Now, let's load sample images from imagenet. The goal is to test how sensitive model predictions are to predefined concepts such as 'striped' when predicting 'zebra' class.
experimental_set_rand = [[canine_concept, random_0_concept], [canine_concept, random_1_concept]]


# Now, let's load sample images from imagenet. The goal is to test how sensitive model predictions are to predefined concepts such as 'striped' when predicting 'zebra' class.
#
# Please, download zebra images and place them under `data/tcav/image/imagenet/zebra` folder in advance before running the cell below. For our experiments we used 50 different zebra images from imagenet dataset.


# Load sample images from folder
husky_imgs = load_image_tensors('siberean_husky', root_path=ROOT_PATH, transform=False)

# Here we perform a transformation and convert the images into tensors, so that we can use them as inputs to NN model.

# Load sample images from folder
husky_tensors = torch.stack([transform(img) for img in husky_imgs])
experimental_set_rand


# zebra class index
# siberean_husky = 250
# white_wolf = 270
canine_ind = 250

tcav_scores_w_random = mytcav.interpret(inputs=husky_tensors,
                                        experimental_sets=experimental_set_rand,
                                        target=canine_ind,
                                        n_steps=5,
                                       )

tcav_scores_w_random


