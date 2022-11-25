import glob
import os
import random
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from captum.concept import Concept
from captum.concept._utils.common import concepts_to_str
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
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
                           recreate_if_exists=False, folder_name=None):
    """
    :param name (str): the general concept encapsulating all the concepts.
                     Eg. "canine"."
    :param concepts (list): list of all the concepts to be assembled.
                     Eg. ["siberean_husky", "white_wolf"].
    :param id: same as id from assemble_concept.
    :param concepts_path: same as concepts_path from assemble_concept.
    :param num_images: number of images that the overall multi_concept should contain (drawn equally from each sub-concept)
                IGNORED IF exists and recreate_if_exists is False
    :param recreate_if_exists: behavior when image folder already exists
    :param folder_name: Use this value to set a custom image folder name (default is concept name)
    """

    # Create a new directory for multi-concept.
    multi_concept_path = os.path.join(concepts_path, name) + "/"
    # If folder name specified, override default name
    if folder_name:
        multi_concept_path = os.path.join(concepts_path, folder_name) + "/"
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
                                          recreate_if_exists: bool, folder_name=None):
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
                                  concepts_path=concepts_path, recreate_if_exists=recreate_if_exists, folder_name=folder_name)


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
            # concepts[node] = assemble_concept(name=node, id=node_data['id'], concepts_path=concepts_path)
            concepts[node] = assemble_multi_concept_from_hierarchy(concept_name=node,
                                                                   h=h,
                                                                   num_images=num_images,
                                                                   concepts_path=concepts_path,
                                                                   recreate_if_exists=recreate_if_exists,
                                                                   folder_name=f"{node}_selected")
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





def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

def plot_tcav_scores(layers, experimental_sets, tcav_scores, outfile='fig.png'):
    fig, ax = plt.subplots(1, len(experimental_sets), figsize=(25, 7))

    barWidth = 1 / (len(experimental_sets[0]) + 1)

    for idx_es, concepts in enumerate(experimental_sets):

        concepts = experimental_sets[idx_es]
        concepts_key = concepts_to_str(concepts)

        pos = [np.arange(len(layers))]
        for i in range(1, len(concepts)):
            pos.append([(x + barWidth) for x in pos[i - 1]])
        _ax = (ax[idx_es] if len(experimental_sets) > 1 else ax)
        for i in range(len(concepts)):
            val = [format_float(scores['sign_count'][i]) for layer, scores in tcav_scores[concepts_key].items()]
            _ax.bar(pos[i], val, width=barWidth, edgecolor='white', label=concepts[i].name)

        # Add xticks on the middle of the group bars
        _ax.set_xlabel('Set {}'.format(str(idx_es)), fontweight='bold', fontsize=16)
        _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])
        _ax.set_xticklabels(layers, fontsize=16)

        # Create legend & Show graphic
        _ax.legend(fontsize=16)

    plt.savefig(outfile)
    plt.show()

def assemble_scores(scores, experimental_sets, idx, score_layer, score_type):
    score_list = []
    for concepts in experimental_sets:
        score_list.append(scores["-".join([str(c.id) for c in concepts])][score_layer][score_type][idx])

    return score_list

def get_pval(scores, experimental_sets, score_layer, score_type, alpha=0.05, print_ret=False):
    P1 = assemble_scores(scores, experimental_sets, 0, score_layer, score_type)
    P2 = assemble_scores(scores, experimental_sets, 1, score_layer, score_type)

    if print_ret:
        print('P1[mean, std]: ', format_float(np.mean(P1)), format_float(np.std(P1)))
        print('P2[mean, std]: ', format_float(np.mean(P2)), format_float(np.std(P2)))

    _, pval = ttest_ind(P1, P2)

    if print_ret:
        print("p-values:", format_float(pval))

    if pval < alpha:  # alpha value is 0.05 or 5%
        relation = "Disjoint"
        if print_ret:
            print("Disjoint")
    else:
        relation = "Overlap"
        if print_ret:
            print("Overlap")

    return P1, P2, format_float(pval), relation

n = 4


def show_boxplots(scores, experimental_sets, n, layer, metric='sign_count', outfile='fig2.png'):
    def format_label_text(experimental_sets):
        concept_id_list = [exp.name if i == 0 else \
                               exp.name.split('_')[0] for i, exp in enumerate(experimental_sets[0])]
        return concept_id_list

    n_plots = 2

    fig, ax = plt.subplots(1, n_plots, figsize=(25, 7 * 1))
    fs = 18
    for i in range(n_plots):
        esl = experimental_sets[i * n: (i + 1) * n]
        P1, P2, pval, relation = get_pval(scores, esl, layer, metric)

        ax[i].set_ylim([0, 1])
        ax[i].set_title(layer + "-" + metric + " (pval=" + str(pval) + " - " + relation + ")", fontsize=fs)
        ax[i].boxplot([P1, P2], showfliers=True)

        ax[i].set_xticklabels(format_label_text(esl), fontsize=fs)
    plt.savefig(outfile)
    plt.show()