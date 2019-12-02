# Readme for the synthetic dataset

1. Design a causal graph using [Tetrad](http://www.phil.cmu.edu/tetrad/) and randomly assign conditional probabilistic tables to each variables, save this model as `OriginalBayesianModel.xml`;
2. Convert the probabilistic model into a deterministic model by randomly select a response for every combination of parents and hidden variables using `synthetic_data_generate.py`.
3. Load the deterministic model in Tetrad and generate a synthetic data, named as `synthetic_data.txt`.
4. Split the `synthetic_data.txt` into a training set `synthetic_train.txt` and a test set `synthetic_test.txt`.