# Quantifying-the-effects-of-social-security-reform

This repo contains the codes of [Quantifying the effects of social security reform](https://drive.google.com/file/d/1YL8Ym6P6LMsPgnHPULJqwaute9DEuP2X/view). The project implements an OLG model in Python. The overview: 

- Overlapping Generations Model (OLG): steady state and transition
- Ex-ante heterogeneity: labor productivity (low and high) and demographic
- Closed economy
- Perfect foresight
- Different pension systems: 

  - Funded defined contribution (FDC)
  - Notional defined contribution (NDC)
  - Non-contributory targeted pension (NTP)
  
- Chilean application: 
  - FDC+NTP
  - Add NDC
  
 **Description of files** 
- `Gen_TSMS.jl`: generate the structure
- `TSMS_functions.jl`: auxiliary functions
- `POIL_PCU_2021.csv`: the data
- `3States_Markov_Switching-OIL-CUP.ipynb`: Jupyter notebook that describes and runs all the code
