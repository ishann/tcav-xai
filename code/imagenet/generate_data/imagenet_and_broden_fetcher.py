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
import shutil
import threading
from pathlib import Path

from hierarchy import Hierarchy

"""
Used to download images from the imagenet dataset and to move concepts from the Broden dataset, rearranging them
in a format that is TCAV readable. Also enables creation of random folders from imagenet

Usage for Imagenet
  imagenet_dataframe = pandas.read_csv("imagenet_url_map.csv")
  fetch_imagenet_class(path="your_path", class_name="zebra", number_of_images=100,
                      imagenet_dataframe=imagenet_dataframe)
                    
Usage for broden:
First, make sure you downloaded and unzipped the broden_224 dataset to a location of your interest. Then, run:
  download_texture_to_working_folder(broden_path="path_were_you_saved_broden", 
                                      saving_path="your_path",
                                      texture_name="striped",
                                       number_of_images=100)
                                      
Usage for making random folders:
  imagenet_dataframe = pandas.read_csv("imagenet_url_map.csv")
  generate_random_folders(working_directory="your_path",
                            random_folder_prefix="random_500",
                            number_of_random_folders=11,
                            number_of_examples_per_folder=100,
                            imagenet_dataframe=imagenet_dataframe)

"""
import pandas as pd
import urllib.request
from PIL import Image
import tensorflow as tf
import urllib
import random
import os
import socket
from joblib import Parallel, delayed
import cv2

kImagenetBaseUrl = "http://imagenet.stanford.edu/api/imagenet.synset.geturls?wnid="
kBrodenTexturesPath = "broden1_224/images/dtd/"
kMinFileSize = 10000

####### Helper functions
""" Makes a dataframe matching imagenet labels with their respective url.

Reads a csv file containing matches between imagenet synids and the url in
which we can fetch them. Appending the synid to kImagenetBaseUrl will fetch all
the URLs of images for a given imagenet label

  Args: path_to_imagenet_classes: String. Points to a csv file matching
          imagenet labels with synids.

  Returns: a pandas dataframe with keys {url: _ , class_name: _ , synid: _}
"""


def make_imagenet_dataframe(path_to_imagenet_classes):
    urls_dataframe = pd.read_csv(path_to_imagenet_classes)
    urls_dataframe["url"] = kImagenetBaseUrl + urls_dataframe["synid"]
    return urls_dataframe


""" Downloads an image.

Downloads and image from a image url provided and saves it under path.
Filters away images that are corrupted or smaller than 10KB

  Args:
    path: Path to the folder where we're saving this image.
    url: url to this image.

  Raises:
    Exception: Propagated from PIL.image.verify()
"""


def download_image(path, url):
    image_name = url.split("/")[-1]
    image_name = image_name.split("?")[0]
    image_prefix = image_name.split(".")[0]
    saving_path = os.path.join(path, image_prefix + ".jpg")
    # urllib.request.urlretrieve(url, saving_path)

    try:
        res = requests.get(url, stream=True, timeout=5)
        if res.status_code == 200:
            with open(saving_path, 'wb') as f:
                shutil.copyfileobj(res.raw, f)
        else:
            raise Exception("skdjflkjds")

        # Throw an exception if the image is unreadable or corrupted
        Image.open(saving_path).verify()

        # Remove images smaller than 10kb, to make sure we are not downloading empty/low quality images
        if tf.io.gfile.stat(saving_path).length < kMinFileSize:
            tf.io.gfile.remove(saving_path)
    # PIL.Image.verify() throws a default exception if it finds a corrupted image.
    except Exception as e:
        tf.io.gfile.remove(
            saving_path
        )  # We need to delete it, since urllib automatically saves them.
        raise e


""" For a imagenet label, fetches all URLs that contain this image, from the main URL contained in the dataframe

  Args:
    imagenet_dataframe: Pandas Dataframe containing the URLs for different
      imagenet classes.
    concept: A string representing Imagenet concept(i.e. "zebra").

  Returns:
    A list containing all urls for the imagenet label. For example
    ["abc.com/image.jpg", "abc.com/image2.jpg", ...]

  Raises:
    tf.errors.NotFoundError: Error occurred when we can't find the imagenet
    concept on the dataframe.
"""

M = {}


def fetch_all_urls_for_concept(imagenet_dataframe, concept):
    if concept in M:
        return M[concept]

    if imagenet_dataframe["class_name"].str.contains(concept).any():
        all_images = imagenet_dataframe[imagenet_dataframe["class_name"] ==
                                        concept]["url"].values[0]
        bytes = urllib.request.urlopen(all_images)
        all_urls = []
        for line in bytes:
            all_urls.append(line.decode("utf-8")[:-2])
        M[concept] = all_urls
        return all_urls
    else:
        raise tf.errors.NotFoundError(
            None, None, "Couldn't find any imagenet concept for " + concept +
                        ". Make sure you're getting a valid concept")


####### Main methods
""" For a given imagenet class, download images from the internet

  Args:
    path: String. Path where we're saving the data. Creates a new folder with
      path/class_name.
    class_name: String representing the name of the imagenet class.
    number_of_images: Integer representing number of images we're getting for
      this example.
    imagenet_dataframe: Dataframe containing the URLs for different imagenet
      classes.

  Raises:
    tf.errors.NotFoundError: Raised when imagenet_dataframe is not provided


"""


def fetch_imagenet_class(path, class_name, number_of_images, imagenet_dataframe):
    if imagenet_dataframe is None:
        raise tf.errors.NotFoundError(
            None, None,
            "Please provide a dataframe containing the imagenet classes. Easiest way to do this is by calling make_imagenet_dataframe()"
        )
    # To speed up imagenet download, we timeout image downloads at 5 seconds.
    socket.setdefaulttimeout(5)

    tf.compat.v1.logging.info("Fetching imagenet data for " + class_name)
    concept_path = os.path.join(path, class_name)
    tf.io.gfile.makedirs(concept_path)
    tf.compat.v1.logging.info("Saving images at " + concept_path)

    # Check to see if this class name exists. Fetch all urls if so.
    all_images = fetch_all_urls_for_concept(imagenet_dataframe, class_name)

    # Fetch number_of_images images or as many as you can.
    num_downloaded = 0
    num_downloaded_lock = threading.Lock()

    def try_download_image(image_url):
        nonlocal num_downloaded, num_downloaded_lock
        if num_downloaded >= number_of_images:
            return
        try:
            download_image(concept_path, image_url)
            with num_downloaded_lock:
                num_downloaded += 1
        except Exception as e:
            pass
            # print("Problem downloading imagenet image. Exception was " + str(e) + " for URL " + image_url)

    # for image_url in all_images:
    #     # if "flickr" not in image_url:
    #     try:
    #         download_image(concept_path, image_url)
    #         num_downloaded += 1
    #
    #     except Exception as e:
    #         pass
    #         # print("Problem downloading imagenet image. Exception was " + str(e) + " for URL " + image_url)
    #     if num_downloaded >= number_of_images:
    #         break
    Parallel(n_jobs=100, backend='threading', verbose=10)(
        delayed(try_download_image)(image_url) for image_url in all_images)

    # If we reached the end, notify the user through the console.
    if num_downloaded < number_of_images:
        print("You requested " + str(number_of_images) +
              " but we were only able to find " +
              str(num_downloaded) +
              " good images from imageNet for concept " + class_name)
    else:
        print("Downloaded " + str(number_of_images) + " for " + class_name)


"""Moves all textures in a downloaded Broden to our working folder.

Assumes that you manually downloaded the broden dataset to broden_path.


  Args:
  broden_path: String.Path where you donwloaded broden.
  saving_path: String.Where we'll save the images. Saves under
    path/texture_name.
  texture_name: String representing DTD texture name i.e striped
  number_of_images: Integer.Number of images to move
"""


def download_texture_to_working_folder(broden_path, saving_path, texture_name,
                                       number_of_images):
    # Create new experiment folder where we're moving the data to
    texture_saving_path = os.path.join(saving_path, texture_name)
    tf.io.gfile.makedirs(texture_saving_path)

    # Get images from broden
    broden_textures_path = os.path.join(broden_path, kBrodenTexturesPath)
    tf.compat.v1.logging.info("Using path " + str(broden_textures_path) + " for texture: " +
                              str(texture_name))
    for root, dirs, files in os.walk(broden_textures_path):
        # Broden contains _color suffixed images. Those shouldn't be used by tcav.
        texture_files = [
            a for a in files if (a.startswith(texture_name) and "color" not in a)
        ]
        number_of_files_for_concept = len(texture_files)
        tf.compat.v1.logging.info("We have " + str(len(texture_files)) +
                                  " images for the concept " + texture_name)

        # Make sure we can fetch as many as the user requested.
        if number_of_images > number_of_files_for_concept:
            raise Exception("Concept " + texture_name + " only contains " +
                            str(number_of_files_for_concept) +
                            " images. You requested " + str(number_of_images))

        # We are only moving data we are guaranteed to have, so no risk for infinite loop here.
        save_number = number_of_images
        while save_number > 0:
            for file in texture_files:
                path_file = os.path.join(root, file)
                texture_saving_path_file = os.path.join(texture_saving_path, file)
                tf.io.gfile.copy(
                    path_file, texture_saving_path_file,
                    overwrite=True)  # change you destination dir
                save_number -= 1
                # Break if we saved all images
                if save_number <= 0:
                    break


""" Creates folders with random examples under working directory.

They will be named with random_folder_prefix as a prefix followed by the number
of the folder. For example, if we have:

    random_folder_prefix = random500
    number_of_random_folders = 3

This function will create 3 folders, all with number_of_examples_per_folder
images on them, like this:
    random500_0
    random500_1
    random500_2


  Args:
    random_folder_prefix: String.The prefix for your folders. For example,
      random500_1, random500_2, ... , random500_n.
    number_of_random_folders: Integer. Number of random folders.
    number of examples_per_folder: Integer. Number of images that will be saved
      per folder.
    imagenet_dataframe: Pandas Dataframe containing the URLs for different
      imagenet classes.
"""


def generate_random_folders(working_directory, random_folder_prefix,
                            number_of_random_folders,
                            number_of_examples_per_folder, imagenet_dataframe):
    socket.setdefaulttimeout(5)

    imagenet_concepts = imagenet_dataframe["class_name"].values.tolist()
    h = Hierarchy()
    for c in h.get_leaf_nodes():
        imagenet_concepts.remove(c)

    for partition_number in range(number_of_random_folders):
        partition_name = random_folder_prefix + "_" + str(partition_number)
        print(f"Making: {partition_name}")
        partition_folder_path = os.path.join(working_directory, partition_name)

        if Path(partition_folder_path).exists():
            continue
        Path(partition_folder_path).mkdir(exist_ok=False, parents=False)

        # Select a random concept
        examples_selected = 0
        examples_selected_lock = threading.Lock()

        def donwload_random_image():
            nonlocal examples_selected, examples_selected_lock
            with examples_selected_lock:
                print(f"count: {examples_selected} of {number_of_examples_per_folder}")
                if examples_selected >= number_of_examples_per_folder:
                    print("Skipping...")
                    return
                examples_selected += 1
            try:
                random_concept = random.choice(imagenet_concepts)
                image_url = random.choice(fetch_all_urls_for_concept(imagenet_dataframe, random_concept))
                download_image(partition_folder_path, image_url)

            except Exception as e:
                with examples_selected_lock:
                    examples_selected -= 1

        while examples_selected < number_of_examples_per_folder:
            try:
                Parallel(n_jobs=100, backend='threading', verbose=100, timeout=30)(
                    delayed(donwload_random_image)() for _ in range(500))
            except Exception as e:
                print(f"Timeout in {partition_name}: {str(e)}")
            finally:
                with examples_selected_lock:
                    examples_selected = len(glob.glob(f"{partition_folder_path}/*.jpg"))
            # for url in urls:
            #     # We are filtering out images from Flickr urls, since several of those were removed
            #     if "flickr" not in url:
            #         try:
            #             download_image(partition_folder_path, url)
            #             examples_selected += 1
            #             if (examples_selected) % 10 == 0:
            #                 tf.compat.v1.logging.info("Downloaded " + str(examples_selected) + "/" +
            #                                           str(number_of_examples_per_folder) + " for " +
            #                                           partition_name)
            #             break  # Break if we successfully downloaded an image
            #         except:
            #             pass  # try new url


##############################################################################################
# Download from ImageNet website not Flickr
##############################################################################################
import tarfile
from tqdm import tqdm
import requests
import glob


def load_wordnet_labels(path='wordnet_labels.txt'):
    labels_to_wordnet = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if ": " not in line:
                continue
            line_arr = line.split(": ")
            wnid = line_arr[0]
            labels = line_arr[1].split(", ")

            for label in labels:
                labels_to_wordnet[label] = wnid
    return labels_to_wordnet


def download_imagenet_classes(class_names, save_path='data', wnet_labels_path='wordnet_labels.txt', num_images=1000):
    labels_to_wnet = load_wordnet_labels(wnet_labels_path)
    make_url = lambda x: f"https://image-net.org/data/winter21_whole/{x}.tar"

    for class_name in tqdm(class_names):
        print(f"Downloading class {class_name} maps to WNET ID {labels_to_wnet[class_name]}")

        if f"{save_path}\\{class_name}.tar" in glob.glob(f"{save_path}/*.tar"):
            continue

        response = requests.get(make_url(labels_to_wnet[class_name]), stream=True)
        if response.status_code == 200:
            with open(f"{save_path}/{class_name}.tar", 'wb') as f:
                f.write(response.raw.read())

        print(f"Extracting tarfile to folder...")
        tarf = tarfile.open(f"{save_path}/{class_name}.tar")
        if len(tarf.getmembers()) < num_images:
            print(
                f"WARNING: Class {class_name} only has {len(tarf.getmembers())} images which is less than {num_images}")
        selected_members = random.sample(tarf.getmembers(), min(len(tarf.getmembers()), num_images))
        for m in selected_members:
            tarf.extract(m, f"{save_path}/{class_name}")
        # tarf.extractall(f"{save_path}/{class_name}")
        try:
            os.remove(f"{save_path}/{class_name}.tar")
        except:
            print("Error deleting file")


if __name__ == '__main__':

    allowed_types = {'jpg', 'jpeg'}

    for path in tqdm(glob.glob("../../data/*")):
        for count, image in enumerate(glob.glob(f"{path}/*")):
            try:
                if not Path(image).is_dir() and "inception5h" not in image and "mobilenet_v2" not in image:
                    img = cv2.imread(image)
                    _ = img.shape
                    img = Image.open(image)
                    img.verify()
                    img = Image.open(image)
                    _ = img.convert("RGB")
                    assert img.format.lower() in allowed_types, f"Image format: {img.format.lower()} not allowed"
                    assert image.lower().endswith(".jpg") or image.lower().endswith(".jpeg"), f"Image {image} has wrong ending"
            except Exception as e:
                print(image, "\t\t", str(e))
                os.remove(image)

            # if not (image.lower().endswith(".jpg") or image.lower().endswith(".jpeg")) and not Path(image).is_dir() and "inception5h" not in image and "mobilenet_v2" not in image:
            #     print(image)
            #     os.remove(image)
