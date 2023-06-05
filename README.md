# Federated Semi-supervised Medical Image Classification via Inter-client Relation Matching
by [Quande Liu](https://github.com/liuquande), [Hongzheng Yang](https://github.com/HongZhengYang), [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou/), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/).

## Introduction

Pytorch implementation for MICCAI 2021 paper "Federated Semi-supervised Medical Image Classification via Inter-client Relation Matching"

![](figure/miccai2021_fedirm.png)
## Usage
1. Clone the repository and go to the repository root directory:

       git clone https://github.com/ysntrkc/FedIRM.git
       cd FedIRM

2. Create a conda environment and activate it:

       conda env create -f environment.yml
       conda activate fedIRM

3. Download datasets from following sources:
   1. [ISIC 2019 (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/isic-2019)
   2. [HAM10000 (Kaggle)](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
   3. [RSNA Intracranial Hemorrhage Detection (drive)](https://drive.google.com/drive/folders/1bhe_0KvdxEli7-6ZrQ9ahaDPpSnvF4UW?usp=share_link)

4. Extract the datasets to `data/` folder and place the content into folders like this:
   1. ISIC 2019: `data/isic2019/`
   2. HAM10000: `data/ham10000/`
   3. RSNA Intracranial Hemorrhage Detection: `data/rsna/`

5. Go to data folder from terminal with `cd data` and run `python prepare_data.py` to prepare the datasets.

6. train the model

       python train_main.py
## Citation

If this repository is useful for your research, please cite:

    @article{liu2021federated,
      title={Federated Semi-supervised Medical Image Classification via Inter-client Relation Matching},
      author={Liu, Quande and Yang, Hongzheng and Dou, Qi and Heng, Pheng-Ann},
      journal={International Conference on Medical Image Computing and Computer Assisted Intervention},
      year={2021}
    }

### Questions

Please contact 'qdliu0226@gmail.com' or 'hzyang05@gmail.com'
