import numpy as np
import os, glob, shutil

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

import ipdb


### Defining image related transformations and functions
# Method to normalize an image to Imagenet mean and standard deviation
def transform(img):

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )(img)


# Now let's define a few helper functions.
def get_tensor_from_filename(filename):
    img = Image.open(filename).convert("RGB")
    return transform(img)


def load_image_tensors(class_name, root_path='data/tcav/image/imagenet/', transform=True):

    #ipdb.set_trace()

    path = os.path.join(root_path, class_name)
    filenames = glob.glob(path + '/*.JPEG')

    tensors = []
    for filename in filenames:
        img = Image.open(filename).convert('RGB')
        tensors.append(transform(img) if transform else img)

    return tensors


# Defining a helper function to load predefined concepts. `assemble_concept` function reads the concepts using a directory path where the concepts are residing and constructs concept object.
def assemble_concept(name, id, concepts_path="data/tcav/image/concepts/"):
    concept_path = os.path.join(concepts_path, name) + "/"
    dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
    concept_iter = dataset_to_dataloader(dataset)

    return Concept(id=id, name=name, data_iter=concept_iter)


# Creates a single multi-concept from a list of concepts.
def assemble_multi_concept(multi_concept, concepts, id, concepts_path="/data/tcav/image/concepts/"):
    """
    multi_concept (str): the general concept encapsulating all the concepts.
                         Eg. "canine"."
    concepts (list): list of all the concepts to be assembled.
                     Eg. ["siberean_husky", "white_wolf"].
    id: same as id from assemble_concept.
    concepts_path: same as concepts_path from assemble_concept.
    """

    # Create a new directory for multi-concept.
    multi_concept_path = os.path.join(concepts_path, multi_concept) + "/"
    """
    import errno
    if not os.path.exists(multi_concept_path):
        try:
            os.makedirs(multi_concept_path, 0o700)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    """
    os.makedirs(multi_concept_path, exist_ok=True)

    # Copy all images from concepts_paths to multi_concept_path.
    #concept_img_paths = []
    for concept in concepts:
        concept_path = os.path.join(concepts_path, concept)
        concept_img_names = os.listdir(concept_path)

        for concept_img_name in concept_img_names:
            src = os.path.join(concept_path, concept_img_name)
            dst = os.path.join(multi_concept_path, concept_img_name)
            shutil.copyfile(src, dst)

    #ipdb.set_trace()
    dataset = CustomIterableDataset(get_tensor_from_filename, multi_concept_path)
    concept_iter = dataset_to_dataloader(dataset)

    return Concept(id=id, name=multi_concept, data_iter=concept_iter)


