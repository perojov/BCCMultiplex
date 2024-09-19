# BCCMultiplex

Code accompanying the paper "Bayesian Consensus Clustering in Multiplex Networks."

- **`GenerativeModels.py`**  
  This script contains the **generative models** for the graphs used in the paper. It implements the models that generate synthetic multiplex networks.

- **`GibbsSamplers.py`**  
  This script provides the **Gibbs sampling algorithms** for posterior inference in the Bayesian consensus clustering framework.

- **`Data/`**  
  The `Data` directory contains all the **network data** used for the experiments. It includes both synthetic and real-world network datasets.

- **`Applications/`**  
  The `Applications` directory demonstrates **how to run inference** on real-world networks. These scripts provide example applications of the methods on specific datasets.
