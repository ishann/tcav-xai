# Human-friendly Compositional Explanations with multi-CAV

## Quick Start
### Install dependencies
1. `cd` to the `code/` directory.
2. Install dependencies with `pip install -r requirements.txt` in order to run our code.

### Running Experiments
- To run experiments with the ImageNet dataset, run `code/imagenet/tcav.py` or `code/imagenet/experiments_in_correctness.ipynb`
- To run experiments with the COMPAS dataset, run `<ADD HERE>`

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
├───<REPLACE>                    # <REPLACE>
├───cifar_noniid_exp             # Data on experiments using nonIID datasets among devices
├───cifar_num_share_exp          # Data on experiments using cifar10 with num_share_devices as the variable
├───cifar_strat_compare          # Data on experiments testing random vs distance based communication strategy
├───figures                      # Directory containing generated graphs and charts
└───mnist_num_share_exp          # Data on experiments using mnist with num_share_devices as the variable
```