# bcaus_nma
Code for paper "Causal Deep Learning on Real-world Data Reveals 
the Comparative Effectiveness of Anti-hyperglycemic 
Treatments".

### Steps for running notebooks locally 
(assuming a conda installation is available)
- `git clone https://github.com/gstef80/bcaus_nma.git`
- `conda env create -f environment.yml` (will create a conda environment named 'bcaus_nma')
- `jupyter notebook` (will start a Jupyter notebook server)

Our method is based on a novel neural network for IPTW ATE estimations 
(BCAUS) and a network metanalysis (NMA) that combines pair-wise 
observations of control-treatment experiments (ATEs estimated using BCAUS). 
Our NMA implementation is validated using R's `netmeta` package 
(see `NMA_demo.ipynb`).

Our BCAUS method (paper accepted at BMC) is demoed in `BCAUS_demo.ipynb`.
