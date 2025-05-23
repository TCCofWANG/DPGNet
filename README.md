# DPGNet

A pytorch implementation for the paper 'DPGNet: A Dynamic Graph Prediction Network for Spatiotemporal Forecasting'. 

# 🎯Overview
![Figure1](./image/framework.jpg)

The overall architecture of the proposed DPGNet

# 📊Regular Prediction
![Table2](./image/Regular_Prediction.png)

Regulat Prediction results. We conduct experiments on five commonly used datasets. For METR-LA, PEMS08, and PEMS-Bay, we set the input length $L$ to 12, with prediction lengths $O$ of 3, 6, and 12, corresponding to time horizons of 15, 30, and 60 minutes, respectively. For the Electricity and Weather datasets, we set $L$ to 168, with prediction lengths $O$ of 3, 6, and 12. For Weather, these correspond to 30, 60, and 120 minutes, while for Electricity, they represent 3, 6, and 12 hours.

# 📊Long-Term Prediction
![Table2](./image/LongTerm_Prediction.png)

Long-Term Prediction results. we compare DPGNet's long-term forecasting performance against baseline models. For METR-LA and PEMS08, the input length $L$ is set to 12, and prediction lengths $O$ are set to 24, 36, and 48, corresponding to 120, 180, and 240 minutes.


# 📊Adaptive Graph Learner replacement experiments
![Table3](./image/AGL_replacement_exp.png)

Adaptive Graph Learner replacement experiments' results.


# 📝Install dependecies
Install the required packages with following code.

```pip install -r requirements.txt```

# 📚Data Preparation
![Tabel4](./image/dataset_intro.png)

Statistics of datasets.

To prepare the benchmark datasets, you can obtain the Los Angeles traffic speed files (METR-LA) and the Bay Area traffice flow data files(PEMS-BAY) from the [DCRNN repository](https://github.com/liyaguang/DCRNN), the Los Angeles traffic flow files(PEMS08) from the [STSGCN repository](https://github.com/Davidham3/STSGCN) and you can obtain the two Time-Series datasets from [FiLM repository](https://github.com/tianzhou2011/FiLM)


# 🚀Run Experiment
We have provided all the experimental scripts for the benchmarks in the `./scripts folder`, which cover all the benchmarking experiments. To reproduce the results, you can run the following shell code.

``` ./scripts/train.sh```