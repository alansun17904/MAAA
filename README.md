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


## Run Experiments


## Development

Installing additional packages:

1. Install via `pip`
2. Find the version used
3. Manually add the package to `environment.yml` with the version. **DO NOT USE conda env export**
