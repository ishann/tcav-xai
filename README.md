# Human-friendly Compositional Explanations with multi-CAV

## Quick Start
### Install dependencies
1. `cd` to the `code/` directory.
2. Install dependencies with `pip install -r requirements.txt` in order to run our code.

### Running Experiments
- To run experiments with the ImageNet dataset, run `code/imagenet/tcav.py` or `code/imagenet/experiments_in_correctness.ipynb`
- To run experiments with the COMPAS dataset, run `code/compas/main.py` or `code/compas/main_unaware.py`

## Important Files

### ImageNet
```
imagenet/
├── generate_data/                  # Directory containing scripts used to define the hierarchy and create concept/random datasets
├── tcav.py                         # Main script used to generate hierarchical explanations for all 16 leaf node datasets
├── results/                        # Contains text files with hierarchical explanations per leaf node dataset.
├── results_latex/                  # The same results as above but with latex tables used in the report
├── HierarchicalExplanation.py      # Module which traces hierarchy, computes tcav scores, and generates explanations
└── utils.py                        # Auxillary functions used throughout code
```

### COMPAS
```
compas/
├── generate_concepts.py            # Generates concepts from the validation data
├── nets_and_datasets.py            # Classes for defining NeuralNetwork and COMPAS Dataset classes and pre-processing functions
├── train.py                        # Training script for vanilla neural network
├── train_unaware.py                # Training script for attribute-unaware neural network
├── main.py                         # TCAV script for vanilla neural network
├── main_unaware.py                 # TCAV script for unaware neural network
├── assess_nnet.py                  # Exploratory script for generating statistics on trained neural network
├── utils.py                        # Utilities for driving both TCAV scripts
└── xai.py                          # Explanations for both TCAV scripts.
```

### Related Work:
The code used to generate CAVs is based on a tutorial from the PyTorch captum repo available at: https://github.com/pytorch/captum/blob/master/tutorials/TCAV_Image.ipynb
