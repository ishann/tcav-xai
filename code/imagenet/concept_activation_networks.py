"""
Concept Activation Networks:
--> Hierarchical networks trained with linear embeddings.
--> Hierarchy may be
    -- pre-defined (say, ImageNet) or
    -- learnt from the data (say, CelebA)
--> TCAV will consume this network instead of GoogleNet with the
    appropriate embeddings only. This will be done in eval mode.

NOTES:
--> Don't write nice code. Write code that works.
--> The network will:
    (a) train linear embeddings for the internal nodes in the hierarchy.
    (b) not train embeddings for the leaf nodes in the hierarchy.
    (c) Be initialized with a train_mode which decides which layers will be
        trained and which will be frozen.

TODOs:
--> Consider replacing linear embeddings with ResNet blocks.
--> Write an actual data-loader with mini-batch training.
--> Refactor to remove the bazillion if statements.
--> Introduce end-to-end training instead of stage-wise training.
"""

import torch
import torch.nn as nn

import torchvision

import ipdb

class TCAN(nn.Module):
    """
    Hierarchical embeddings learnt based on the FC layer of a pre-trained
    networks.
    See hierarchy at ./generate_data/hierarchy.json.
    Stage-wise training is currently implemented.
    Because this is a class project and we have limited bandwidth.
    """

    def __init__(self, train_mode="entity", positive_concepts):
        """
        train_mode (str): tells us which layers will be frozen and
                          which will be learnt.
        positive_concepts (list): tells us which concepts will form the
                                  positive samples in train and val sets.
        """
        super(TCAN, self).__init__()

        # TODO: Make backbone a parameter to allow ResNet experiments.
        self.backbone = torchvision.models.googlenet(pretrained=True)

        # Trained using all 16 concepts vs. random.
        self.embed_entity = nn.Linear(1000,128)

        # Trained using 6 concepts vs. random.
        self.embed_organism = nn.Linear(128, 128)

        # Trained using 4 concepts vs. random.
        self.embed_canine = nn.Linear(128, 128)

        # Trained using 1 concepts vs random.
        # Wrong. Leaf nodes are not trained; only tested with TCAV.
        # self.embed_white_wolf = nn.Linear(128, 128)

        # Trained using 3 concepts vs. random.
        self.embed_dog = nn.Linear(128, 128)

        # Trained using 1 concept vs. random.
        # Wrong. Don't train embeddings for leaf nodes. Only test with TCAV.
        # self.embed_dalmatian = nn.Linear(128, 128)
        # self.embed_siberean_husky = nn.Linear(128, 128)
        # self.embed_golden_retriever = nn.Linear(128, 128)

        self.relu = nn.ReLU(inplace=True)

        self.train_mode = train_mode

        if self.train_mode == "entity":
            pass
            # TODO: Freeze backbone only.
            # TODO: Assert that positive samples will include 16 concepts.

        elif self.train_mode == "organism":
            pass
            # TODO: Freeze backbone + entity.
            # TODO: Assert that positive concepts will include 6 concepts.

        elif self.train_mode == "canine":
            pass
            # TODO: Freeze backbone + entity + organism.
            # Assert that positive concepts will include 4 concepts.

        elif self.train_mode == "dog":
            pass
            # TODO: Freeze backbone + entity + organism + canine.
            # Assert that positive concepts will include 3 concepts.

        else:
            raise ValueError("Unknown train_mode encountered.")

    def _forward(self, x):

        # This occurs for all train_mode values.
        return self.relu(self.backbone(x))


    def forward(self, x):

        # TODO: Make sure logic for freezing all prior embeddings is
        #       in place using requires_grad=False.

        x = self._forward(x)

        # TODO: There should be a more elegant way to write this...
        if self.train_mode=="entity":
            x_entity = self.relu(self.embed_entity(x))
            return x_entity

        elif self.train_mode=="organism":
            x_entity  = self.relu(self.embed_entity(x))
            x_organism = self.relu(self.embed_organism(x_entity))
            return x_organism

        elif self.train_mode=="canine":
            x_entity = self.relu(self.embed_entity(x))
            x_organism = self.relu(self.embed_organism(x_entity))
            x_canine = self.reul(self.embed_canine(x_organism))
            return x_canine

        elif self.train_mode=="dog":
            x_entity = self.relu(self.embed_entity(x))
            x_organism = self.relu(self.embed_organism(x_entity))
            x_canine = self.reul(self.embed_canine(x_organism))
            x_dog = self.relu(self.embed_dog(x_canine))
            return x_dog

        else:
            # Probably don't need this because __init__ will check for this.
            raise ValueError("Unknown train_mode encountered.")


# TODO: Initialize CAN.

# TODO: Hyper-parameters.

# TODO: Dataloader.

# TODO: Optimizer.

# TODO: Training loop.

# TODO: Grid search hyper-params?












