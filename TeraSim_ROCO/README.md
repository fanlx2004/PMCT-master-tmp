# Experiments on TeraSim and ROCO

# Introduction 

## About

The folder `TeraSim_ROCO` contains the source code and data for the latest experiments of PMCT. We apply PMCT for close-loop validation on the simulator TeraSim [1] and open_loop evaluation on the real-world conflict dataset ROCO [2].

## Code Structure

```
TeraSim_ROCO/
├─ requirements.txt : required packages
├─ setup_environment.sh : setup tool for TeraSim
├─ ROCO_prediction.py : main function for experiments on ROCO
├─ possess_data.py : function for possess the raw data generated from TeraSim
├─ configs/ : configs for running TeraSim
├─ data/ : data that is used in PMCT experiments
│  ├─ ROCO_collision_data/ : possessed collision data from the ROCO dataset
│  ├─ ROC_PR_data/ : data for ploting ROC and PR curves
│  ├─ provability_data/ : data of provability experiment
│  └─ terasim_data/ : data for network training and verification
├─ examples/ : example traffic systems in TeraSim
├─ figures/  : figures that PMCT experiments generate
│  ├─ ROCO_figure/ : ROCO prediction from figure
│  ├─ TeraSim_cross_figure/ : result figures from intersection model 
│  └─ TeraSim_road_figure/ : result figures from road model 
├─ model_params/ : PMCT network parameters
├─ packages/ : Terasim tools
├─ scripts/ : Terasim running function
└─ tools/ : tools for network structure and metric calculation
```

# Installation

## Pre-requirements
  - Python installation
    - This repository is developed and tested under python 3.10.19 on Ubuntu 20.04.6 system (conda version: 24.11.3).

## Installation and configuration

### Create a new virtual environment 

```bash
cd TeraSim_ROCO
conda create -n myenv python=3.10 -y
conda activate myenv
```

### Install all required packages

Install all required Python packages and dependencies and set up TeraSim environment.

```bash
./setup_environment.sh
pip install -r requirements.txt
```

<!-- USAGE EXAMPLES -->

# Usage

The folder includes code (network training and verification, experiments) and experiment outcomes (net paramaters, datas and figures). We construct the Q-network by training and verifying, conduct close-loop experiments of PMCT probability and metrics comparison on TeraSim-generated data, and carry out open-loop experiments on ROCO dataset. Several main functions are provided for the above usages. 

## Usage 1 (Network training and verification) (Optional)

The trained road model and the trained intersection model for PMCT are saved in `./model_params/`. To obtain a new version of network paramaters, generate data from TeraSim first with the following command:

```bash
python scripts/run_experiments_debug.py
```

The number of data to be generated can be set in `scripts/run_experiments_debug.py`. We recommend generating 100000 samples.

After generating, possess the data by running:

```bash
python possess_data.py
```

Then, you can train and verify the network by running:
```bash
python training_and_verification_main.py --mode road
python training_and_verification_main.py --mode cross
```

The model under training are saved in `./model_params/`.

## Usage 2 (close-loop experiments on TeraSim)

The function to calculate PMCT based on an inputed initial state is defined in `./tools/metric_solver.py`. We provide main functions to repeat the close-loop experiments shown in our paper.

### Usage 2.1 (Experiments on the provability of PMCT)

To conduct the experiments on verifying the provability of PMCT through TeraSim simulations, run the following command:
```bash
python provability_experiment.py
```

The results for road model or intersection model can be switched by setting the varible `accident_mode`. If you want to re-generate the true collision times and predicted values used in the experiments, set the boolean varible `do_calculate` to `False` in Line 12 of `provability_experiment.py`.

### Usage 2.2 (Experiments on the safety metrics performance comparison)

We compare the performance in real-time safety assessment between the proposed PMCT and other safety metrics through scenario simulations. The functions to calculate other function are in `./tools/metric_solver.py`.

Run the following command to repeat the experiment and draw the ROC and PR curves:
```bash
python ROC_PR.py
```

The results for road model or intersection model can be switched by setting the varible `accident_mode`. If you want to re-generate the true collision times and predicted values used in the experiments, set the boolean varible `do_calculate` to `False` in Line 23 of `ROC_PR.py`. However, due to the superior time consumption calculating MPrISM, we suggest using the data we saved.

## Usage 3 (open-loop experiments on ROCO dataset)

We employ PMCT to predict collision on the collision data of ROCO dataset. The possessed version of three collision trajectories are saved in `./data/ROCO_collision_data/`. We provide main functions to repeat the open-loop experiments shown in our paper.

Run the following command to repeat the experiment and draw the prediction results:
```bash
python ROCO_prediction.py 
```

If you want to re-generate the true collision times and predicted values used in the experiments, set the boolean varible `do_calculate` to `False` in Line 11 of `ROC_PR.py`. However, due to the superior time consumption calculating MPrISM, we suggest using the data we saved.


# Reference

[1] Sun, H., Yan, X., Qiao, Z., Zhu, H., Sun, Y., Wang, J., Shen, S., Hogue, D., Ananta, R., Johnson, D., et al., 2025. Terasim: Uncovering unknown unsafe events for autonomous vehicles through generative simulation. arXiv preprint arXiv:2503.03629.

[2] Meng, D., Sayer, O., Zhang, R., Shen, S., Li, H., Liu, H.X., 2023. Roco: A roundabout traffic conflict dataset. arXiv preprint arXiv:2303.00563.

# Developers

Lingxiang Fan (fanlx23@mails.tsinghua.edu.cn)

Linxuan He (hlx24@mails.tsinghua.edu.cn)

For help or issues using the code, please create an issue for this repository or contact Lingxiang Fan (fanlx23@mails.tsinghua.edu.cn).

# Contact

For general questions about the paper, please contact Shuo Feng (fshuo@tsinghua.edu.cn).