# MAAA

This is the MAAA code repo.

## Setup

### Install Conda

Install conda (skip if already installed).

1. Run `mkdir -p ~/miniconda3` in root
2. Run `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh`
3. Run `bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3`
4. Run `rm ~/miniconda3/miniconda.sh`
5. Run `source ~/.bashrc`


# Setup Environment

1. Run the following command: `conda env create -f environment.yml`
2. Run `conda activate MAAA_CD`
3. Unzip data.zip


## Run Experiments

Pruning scripts are found in scripts/ - following the same convention as the EdgePruning codebase. Currently, they are not configured to a specific experiment, rather just testing basic functionality. 


Evaluation scripts are found in `src/layer2/eval/{ioi/gt/gp}.py`. To evaluate a circuit found with the default settings above, you can simply run, for example,
```
python src/eval/ioi.py -m /path/to/pruned_model -w
```


## Development

We recommend developing on an evironment that has cuda available. The reason for this is that the scripts will often halt because cuda is not available, before testing the majority of other functionality / configuration. For this reason, any environment with a compatible version of CUDA (see Pytorch for compatability), regardless of VRAM / hardware capabilities, is recommended for development. Additionally, linux makes things easier (but should not be necessary).

Installing additional packages:

1. Install via `pip`
2. Find the version used
3. Manually add the package to `environment.yml` with the version. **DO NOT USE conda env export**

## Todo:

1. Get dataset (see src/layer3/test.ipynb - with training_dataset)
2. Compute_loss in src/layer3/l3mezo.py - make the loss kl divergence + lambda * output of func
3. Create function which computes the difference in weights. 