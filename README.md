# bcaus_nma
Code and tutorials for our propensity-based causal inference methods designed to estimate treatment effects for masssively multi-arm studies using real-world data

The core methodology is a novel neural network approach which uses a custom loss function to explictly balance covariates in addition to specifying propentisty for IPTW ATE estimations and is described in the manuscript entitled: *"Minimizing Bias in Massive Multi-Arm Observational Studies with BCAUS: Balancing Covariates Automatically Using Supervision"* . The code can be found in `bcaus.py` and a demonstration of it's usage can be found in `BCAUS_demo.ipynb`
 
We've extended the core methodology by including a network metanalysis (NMA) after BCAUS that combines pair-wise observations of control-treatment experiments to leverage both direct and indirect comparisons. Our NMA implementation is validated using R's `netmeta` package (see `NMA_demo.ipynb`). This enhanced approached is what was used in the manuscript entitled *"Causal Deep Learning on Real-world Data Reveals the Comparative Effectiveness of Anti-hyperglycemic Treatments"*.

### Steps for running notebooks locally 
(assuming a conda installation is available)
- `git clone https://github.com/gstef80/bcaus_nma.git`
- `conda env create -f environment.yml` (will create a conda environment named 'bcaus_nma')
- `jupyter notebook` (will start a Jupyter notebook server)

 


