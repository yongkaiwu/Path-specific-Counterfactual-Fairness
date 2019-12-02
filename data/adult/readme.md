# Readme for the adult dataset

1. Obtain the causal model `adult.xml` from the IJCAI 2019 paper "Counterfactual Fairness: Unidentification, Bound and Algorithm". The pre-processing in IJCAI 2019 paper is summarized as follows.
    1. Get data from [UCI repository](http://mlr.cs.umass.edu/ml/datasets/Adult);
    2. Manipulation:
        1. save data and test as csv file,
        2. add attribute names to csv file,
        3. combine data and test data,
    3. Convert the data to binary using `adult_processing.py`.
2. Run `adult_data_CE.py` to compute the bounds of counterfactual fairness.
3. Experiment result:
        || IJCAI 19      | our method    |
        || lb      ub    | lb      ub    |
        0  0.0541  0.2946  0.1498  0.1944
        1 -0.1314  0.1091 -0.1314  0.1091
        2  0.1878  0.3210  0.2507  0.2890
        3 -0.0356  0.0976 -0.0356  0.0976
        4  0.1676  0.5289  0.4419  0.5289
        5 -0.1634  0.1979 -0.0731  0.1979
        6  0.1290  0.4689  0.3942  0.4689
        7 -0.1808  0.1591  0.0014  0.1591