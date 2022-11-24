import glob
import os
import random
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from captum.concept import Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from torchvision import transforms
from tqdm import tqdm

from generate_data.hierarchy import Hierarchy


# ..........torch imports............
# .... Captum imports..................


# import ipdb


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


def load_image_tensors(class_name, root_path='data/tcav/image/imagenet/', transform=True, count=100):
    # ipdb.set_trace()

    path = os.path.join(root_path, class_name)
    filenames = glob.glob(path + '/*.JPEG')

    tensors = []
    for i, filename in enumerate(filenames):
        if i >= count:
            break

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
def assemble_multi_concept(name, concepts, id, num_images, concepts_path="/data/tcav/image/concepts/",
                           recreate_if_exists=False):
    """
    name (str): the general concept encapsulating all the concepts.
                     Eg. "canine"."
    concepts (list): list of all the concepts to be assembled.
                     Eg. ["siberean_husky", "white_wolf"].
    id: same as id from assemble_concept.
    concepts_path: same as concepts_path from assemble_concept.
    num_images: number of images that the overall multi_concept should contain (drawn equally from each sub-concept)
                IGNORED IF exists and recreate_if_exists is False
    recreate_if_exists: behavior when image folder already exists
    """

    # Create a new directory for multi-concept.
    multi_concept_path = os.path.join(concepts_path, name) + "/"
    """
    import errno
    if not os.path.exists(multi_concept_path):
        try:
            os.makedirs(multi_concept_path, 0o700)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    """
    delete_old = Path(multi_concept_path).exists() and recreate_if_exists
    copy_files = recreate_if_exists or not Path(multi_concept_path).exists()

    # Delete directory if exists and recreate
    if delete_old:
        assert len(os.listdir(multi_concept_path)) == len(
            glob.glob(os.path.join(multi_concept_path, "*.JPEG"))), "Directory to delete must only contain images"
        shutil.rmtree(Path(multi_concept_path), ignore_errors=True)

    if copy_files:
        Path(multi_concept_path).mkdir(exist_ok=True)

        # Copy all images from concepts_paths to multi_concept_path.
        num_images_from_concepts = [len(arr) for arr in np.array_split(range(num_images), len(concepts))]

        for concept, num_images_from_concept in zip(concepts, num_images_from_concepts):
            concept_path = os.path.join(concepts_path, concept)
            concept_img_names = os.listdir(concept_path)
            random.shuffle(concept_img_names)  # Re-order list

            assert num_images_from_concept <= len(
                concept_img_names), "Number of images drawn from concept must be at most the number of available images"

            for idx, concept_img_name in enumerate(concept_img_names):
                src = os.path.join(concept_path, concept_img_name)
                dst = os.path.join(multi_concept_path, concept_img_name)
                shutil.copyfile(src, dst)

                # Only copy desired amount
                if idx + 1 >= num_images_from_concept:
                    break

    # ipdb.set_trace()
    dataset = CustomIterableDataset(get_tensor_from_filename, multi_concept_path)
    concept_iter = dataset_to_dataloader(dataset)

    return Concept(id=id, name=name, data_iter=concept_iter)


def assemble_multi_concept_from_hierarchy(concept_name: str, h: Hierarchy, num_images: int, concepts_path: str,
                                          recreate_if_exists: bool):
    """
    :param concept_name: Name of the concept (must be in hierarchy.json)
    :param h: hierarchy object
    :param recreate_if_exists: behavior if folder exists
    :param concepts_path:
    :param num_images: num images in non-leaf folders
    """
    node = h.find_node(concept_name)
    assert node is not None, "Couldn't find concept name in hierarchy"

    leaf_nodes = h.get_leaf_nodes(of=concept_name)
    return assemble_multi_concept(name=concept_name, concepts=leaf_nodes, id=node['id'], num_images=num_images,
                                  concepts_path=concepts_path, recreate_if_exists=recreate_if_exists)


def assemble_all_concepts_from_hierarchy(h: Hierarchy, num_images: int, concepts_path: str, recreate_if_exists: bool):
    """
    Assemble all concepts in hierarchy
    :param h: Hierarchy object
    :param num_images: Number of images contained in each ***non-leaf*** concept
    :param concepts_path: Directory where concept folders will be created and contains leaf node images
    :return: dict mapping concept name to concept object
    """

    nodes = h.get_all_nodes()
    leaf_nodes = set(h.get_leaf_nodes())

    concepts = dict()

    print("Assembling concepts...")
    for node in tqdm(nodes, desc="Creating concepts"):
        node_data = h.find_node(node)

        if node in leaf_nodes:
            concepts[node] = assemble_concept(name=node, id=node_data['id'], concepts_path=concepts_path)
        else:
            concepts[node] = assemble_multi_concept_from_hierarchy(concept_name=node, h=h, num_images=num_images,
                                                                   concepts_path=concepts_path,
                                                                   recreate_if_exists=recreate_if_exists)
    return concepts


def assemble_random_concepts(concepts_path: str):
    random_concepts = dict()
    random_dirs = glob.glob(os.path.join(concepts_path, "random_*", ""))

    for random_dir in tqdm(random_dirs, desc="Creating Random concepts"):
        idx = int(re.search(r"random_([\d]+)", random_dir).group(1))
        random_concepts[idx] = assemble_concept(name=f"random_{idx}", id=2000 + idx, concepts_path=concepts_path)

    return random_concepts


def generate_experiments(concepts_dict: dict, random_concepts_dict: dict):
    experiments = []
    for i, concept in enumerate(concepts_dict):
        experiments.append([concepts_dict[concept], random_concepts_dict[i]])
    return experiments
