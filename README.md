# Code for "Asynchronous Upper Confidence Bound Algorithms for Federated Linear Bandits"

This repository contains implementation of the proposed algorithms Async-LinUCB, Async-LinUCB-AM, and baseline algorithm Sync-LinUCB for comparison.
For experiments on the synthetic dataset, run:
```console
python SimulationHomogeneousClients.py --T 50000 --N 1000  # simulate homogeneous clients
python SimulationHeterogeneousClients.py --T 50000 --N 1000 --globaldim 16  # simulate heterogeneous clients
```

Experiment results can be found in "./SimulationResults/" folder, which contains:
- "regretAndcommCost\_[startTime].png": plot of accumulated regret / communication cost over time for each algorithm
- "AccRegret\_[startTime].csv": regret at each iteration for each algorithm
- "AccCommCost\_[startTime].csv": communication cost at each iteration for each algorithm
- "ParameterEstimation\_[startTime].csv": l2 norm between estimated and ground-truth parameter at each iteration for each algorithm

For experiments on realworld dataset, e.g. LastFM, Delicious, MovieLens, 
- First download these publicly available data into Dataset folder 
- Then process the dataset following instructions in the appendix of the paper, which would generate the item feature vector file and the event file. Example scripts for processing data are given in Dataset folder.
- Run experiments using the provided python script, e.g. ``python SimulationRealworldData.py --dataset LastFM``


Updates::
For Testing LinGapE in homogeneous clients, run:
```console
python SimHomogeneousClientsLinGapE.py
```

Experiment results can be found in "./LinGapE_Simulations" folder, which contains:
- "AccCommCost\_[startTime].csv": communication cost at each iteration for each algorithm
- "SampleComplex\_[startTime].csv": number of samples required to estimate the best for each algorithm