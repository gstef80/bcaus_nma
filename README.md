# bcaus_nma
Code for paper "Causal Deep Learning on Real-world Data Reveals 
the Comparative Effectiveness of Anti-hyperglycemic 
Treatments".

Our method is based on a novel neural network for IPTW ATE estimations 
(BCAUS) and a network metanalysis (NMA) that combines pair-wise 
observations of control-treatment experiments (ATEs estimated using BCAUS). 
Our NMA implementation is validated using R's `netmeta` package 
(see `NMA_demo.ipynb`).
