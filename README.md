# bcaus_nma
Code and tutorials for our propensity-based causal inference methods 
designed to estimate treatment effects for massively multi-arm studies 
using real-world data.

The core methodology is a novel neural network approach which uses 
a custom loss function to explicitly balance covariates in addition 
to specifying propensity for IPTW ATE estimations and is described 
in the manuscript entitled: 
*"Minimizing Bias in Massive Multi-Arm Observational Studies with BCAUS: 
Balancing Covariates Automatically Using Supervision"*. 
The code can be found in `bcaus.py` and a demonstration of its usage 
can be found in `BCAUS_demo.ipynb`
 
We've extended the core methodology by including a network meta-analysis 
(NMA) after BCAUS that combines pair-wise observations of 
control-treatment experiments to leverage both direct and indirect 
comparisons. Our NMA implementation is validated using R's `netmeta` 
package (see `NMA_demo.ipynb`). This enhanced approached is what was 
used in the manuscript entitled 
*"Causal Deep Learning on Real-world Data Reveals the Comparative 
Effectiveness of Anti-hyperglycemic Treatments"* 
(henceforth referred to as *paper*)

### Steps for running notebooks locally 
(assuming a conda installation is available)
- `git clone https://github.com/gstef80/bcaus_nma.git`
- `conda env create -f environment.yml` (will create a conda environment named 'bcaus_nma')
- `jupyter notebook` (will start a Jupyter notebook server)

### Pseudo-code for paper
```python
# 10 SME-defined clinical cohorts
for cohort in clinical_cohorts:
    # treatments with cohort sizes above threshold 
    treatments = cohort_treatments(cohort, threshold=35)
    ate_estimates = []
    for ct in treatments:
        for tx in treatments:
            # perform pair-wise BCAUS experiments
            ate = BCAUS(cohort, ct, tx)
            ate_estimates.append(ate)
    # perform NMA on ATEs
    ranks = NMA(ate_estimates)
```
### Settings for MCMC (NMA)
- trace_samples=100000
- burn_ratio=0.5
- num_chains=4
<br>The choice of these values was based on NICE DSU TECHNICAL SUPPORT 
  DOCUMENT 2 (http://nicedsu.org.uk/wp-content/uploads/2017/05/TSD2-General-meta-analysis-corrected-2Sep2016v2.pdf).
  



