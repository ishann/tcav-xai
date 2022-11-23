"""
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
""" Downloads models and datasets for imagenet

    Content downloaded:
        - Imagenet images for the zebra class.
        - Full Broden dataset(http://netdissect.csail.mit.edu/)
        - Inception 5h model(https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/inception5h.py)
        - Mobilenet V2 model(https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)

    Functionality:
        - Downloads open source models(Inception and Mobilenet)
        - Downloads the zebra class from imagenet, to illustrate a target class
        - Extracts three concepts from the Broden dataset(striped, dotted, zigzagged)
        - Structures the data in a format that can be readily used by TCAV
        - Creates random folders with examples from Imagenet. Those are used by TCAV.

    Example usage:

    python download_and_make_datasets.py --source_dir=YOUR_FOLDER --number_of_images_per_folder=50 --number_of_random_folders=10
"""
import os
import subprocess
from pathlib import Path

from tqdm import tqdm

import imagenet_and_broden_fetcher as fetcher
from hierarchy import Hierarchy


def make_concepts_targets_and_randoms(source_dir,
                                      number_of_images_per_folder,
                                      number_of_random_folders,
                                      make_broden,
                                      make_imagenet,
                                      make_random,
                                      imagenet_classes,
                                      broden_concepts):
    # Run script to download data to source_dir
    print("Run script to download data to source_dir")
    Path(source_dir).mkdir(exist_ok=True, parents=True)

    if make_broden:
        if not Path(os.path.join(source_dir, 'broden1_224/')).exists() or not Path(
                os.path.join(source_dir, 'inception5h')).exists():
            print("Error: broden or inception not found")
            subprocess.call(['bash', 'FetchDataAndModels.sh', source_dir])

    # Determine classes that we will fetch
    # imagenet_classes = ['zebra']
    # broden_concepts = ['striped', 'dotted', 'zigzagged']

    # make targets from imagenet
    if make_imagenet:
        print("make targets from imagenet")
        imagenet_dataframe = fetcher.make_imagenet_dataframe("./imagenet_url_map.csv")
        # for image in tqdm(imagenet_classes):
        #     fetcher.fetch_imagenet_class(source_dir, image, number_of_images_per_folder, imagenet_dataframe)

        fetcher.download_imagenet_classes(imagenet_classes, save_path=source_dir,
                                          num_images=number_of_images_per_folder)

    # Make concepts from broden
    if make_broden:
        print("Make concepts from broden")
        for concept in tqdm(broden_concepts):
            fetcher.download_texture_to_working_folder(broden_path=os.path.join(source_dir, 'broden1_224'),
                                                       saving_path=source_dir,
                                                       texture_name=concept,
                                                       number_of_images=number_of_images_per_folder)

    # Make random folders. If we want to run N random experiments with tcav, we need N+1 folders.
    if make_random:
        print("Make random folders. If we want to run N random experiments with tcav, we need N+1 folders.")
        imagenet_dataframe = fetcher.make_imagenet_dataframe("./imagenet_url_map.csv")
        fetcher.generate_random_folders(
            working_directory=source_dir,
            random_folder_prefix="random",
            number_of_random_folders=number_of_random_folders + 1,
            number_of_examples_per_folder=number_of_images_per_folder,
            imagenet_dataframe=imagenet_dataframe
        )


if __name__ == '__main__':
    source_dir = 'data'

    h = Hierarchy()
    imagenet_classes = h.get_leaf_nodes()
    imagenet_classes = ['volleyball', 'tennis ball', 'soccer ball', 'golf ball']

    # create folder if it doesn't exist
    Path(os.path.join(source_dir)).mkdir(exist_ok=True, parents=True)
    print("Source directory at " + source_dir)

    # Make data
    make_concepts_targets_and_randoms(source_dir=source_dir,
                                      number_of_images_per_folder=1000,
                                      number_of_random_folders=len(h.get_all_nodes()),
                                      make_broden=False,
                                      make_imagenet=False,
                                      make_random=True,
                                      imagenet_classes=imagenet_classes,
                                      broden_concepts=None)
    print("Successfully created data at " + source_dir)
