# Readme

This is an implementation for [PC-Fairness: A Unified Framework for Measuring Causality-based Fairness](http://papers.nips.cc/paper/8601-pc-fairness-a-unified-framework-for-measuring-causality-based-fairness) in NIPS 2019.

## Development Environment

1. Our implementation is based on Python 3.6.6 in Windows 10 (64-Bit).
2. The python distribution [Anaconda](https://www.anaconda.com) or [Miniconda](https://repo.continuum.io/miniconda/) is highly recommended. 
3. Since we utilize the environment management tool `conda`, Miniconda is minimal and sufficient.

## Reproduction

To re-produce this repository:

1. Recover the environment by `conda env create --file environment.yml --name YOUR_ENV_NAME`.
2. Run
   1. `python D1_PC.py` to get Table 2;
   2. `python D2_PE.py` and `python D2_CE` to get Table 3;
   3. `python adult_data_CE.py` to get Table 4.

## Citation

Please cite the original paper if you use this implementation in your manuscript.

```
@inproceedings{DBLP:conf/nips/Wu0WT19,
  author    = {Yongkai Wu and
               Lu Zhang and
               Xintao Wu and
               Hanghang Tong},
  editor    = {Hanna M. Wallach and
               Hugo Larochelle and
               Alina Beygelzimer and
               Florence d'Alch{\'{e}}{-}Buc and
               Emily B. Fox and
               Roman Garnett},
  title     = {PC-Fairness: {A} Unified Framework for Measuring Causality-based Fairness},
  booktitle = {Advances in Neural Information Processing Systems 32: Annual Conference
               on Neural Information Processing Systems 2019, NeurIPS 2019, 8-14
               December 2019, Vancouver, BC, Canada},
  pages     = {3399--3409},
  year      = {2019},
  url       = {http://papers.nips.cc/paper/8601-pc-fairness-a-unified-framework-for-measuring-causality-based-fairness},
  timestamp = {Fri, 06 Mar 2020 16:59:11 +0100},
  biburl    = {https://dblp.org/rec/conf/nips/Wu0WT19.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
