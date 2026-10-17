# Pretraining Respiratory Sound Representations Using Metadata and Contrastive Learning
Authors : Ilyass Moummad, Nicolas Farrugia

-----
Update : Our work is accepted for WASPAA 2023 (poster), we updated this repository to include M-SCL and SPRSound dataset
-----

This is the official pytorch implementation of our work [Pretraining Respiratory Sound Representations using Metadata and Contrastive Learning](https://arxiv.org/abs/2210.16192)

## Dependencies:
Launch : ```pip install -r requirements.txt```

## Dataset:
Put the data files in the data folder

## Pretrained models:
Put the pretrained pth files in the panns folder. \
Here is the link to download the weights for PANNs: https://zenodo.org/record/3987831 \
This code supports CNN6, CNN10 and CNN14 with the corresponding weights "Cnn6_mAP=0.343.pth", "Cnn10_mAP=0.380.pth" and "Cnn14_mAP=0.431.pth"

## Metadata:
Metadata csv file for each dataset are in the corresponding folder of data folder. 

## Training:
```main.py``` launches the training
### Arguments:
```--dataset```: to specificy which dataset to use, possible values are ``ICBHI`` and ``SPRS``. \
If the dataset is SPRS, ```--mode``` specifies whether to use the inter-patient or intra-patient split, possible values are ``inter`` and ``intra``. \
```--method METHOD```: the training method METHOD, ``sl`` for cross entropy, ``scl`` for supervised contrastive, ``hybrid`` for a combination of both, and ``mscl`` for multi-head supervised contrastive incorporating metadata. \
```--backbone BACKBONE```: the backbone to be used, ``cnn6``, ``cnn10`` or ``cnn14``. \
```--scratch```: to train from scratch (when this argument is not encountered, the models are intiailized using AudioSet weights). \
```--lr LR```: the learning rate for training. \
```--bs BS```: the batch size. \
Check ```args.py``` for more arguments.

### Reproducibility:
To reproduce our results, keep all arguments by default value except the learning rate (and the generic arguments such as the number of workers to be used and the desired device to train on). \
To train from scratch, launch the following script : ```python main.py --dataset DATASET --scratch --lr 1e-3 --backbone BACKBONE --method METHOD``` using the desired training method and backbone. \
To train using AudioSet intialization, launch the following script : ```python main.py --dataset DATASET --lr 1e-4 --backbone BACKBONE --method METHOD``` using the desired training method and backbone. \
To train using M-SCL (SCL with 2 heads, one for respiratory classification task and one for metadata task), add the following arguments : ```--metalabel metalabel --lam tradeoff``` using the desired metadata, the code supports sex 's' and age 'a', 'c' for respiratory class, 'sa' use both sex and age for the auxiliary task (default value which we recommend). the tradeoff for the main loss and the auxiliary loss can also be adjusted. 
To obtain the values of the last line of the table below, launch the following script : ```python main.py --dataset DATASET --lr 1e-4 --backbone cnn6 --method scl --mscl --metalabel sa --lam 0.75```

## Quantitative Results
We optimized hyperparameters for CNN6, and we simply report CNN10 from scratch and pretrained CNN14 scores on ICBHI without any hyperparameter tuning. \
We report results over 10 identical runs:

| Backbone | Method |     _Sp_    |     _Se_    |     _Sc_    | # of Params | Ext. Dataset |
|:--------:|:------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
|   Cnn6   |   CE   | 76.72(3.97) | 31.12(3.72) | 53.92(0.71) |     4.3     |       -      |
|          |   SCL  | 76.17(3.84) | 27.97(3.92) | 52.08(1.06) |             |              |
|          | Hybrid | 75.35(5.47) | 33.84(5.67) |  54.74(0.5) |             |              |
|   Cnn10  |   CE   |  73.45(6.7) |  36.8(6.61) | 55.13(1.56) |     4.8     |       -      |
|          |   SCL  | 74.78(5.89) |  30.38(5.5) | 52.59(1.35) |             |              |
|          | Hybrid | 78.12(6.14) | 33.07(5.43) |  55.6(1.13) |             |              |
|   Cnn6   |   CE   | 70.09(3.08) | 40.39(2.97) | 55.24(0.43) |     4.3     |   AudioSet   |
|          |   SCL  | 75.95(2.31) | 39.15(1.89) | 57.55(0.81) |             |              |
|          | Hybrid | 70.47(2.07) | 43.29(1.83) | 56.89(0.55) |             |              |
|   Cnn14  |   CE   | 75.63(4.13) | 38.13(5.07) | 57.32(0.54) |     75.4    |   AudioSet   |
|          |   SCL  |  80.67(4.2) | 32.93(4.37) |  56.92(0.9) |             |              |
|          | Hybrid | 80.73(3.86) | 34.96(3.59) | 57.85(0.48) |             |              |
|   Cnn6   |  M-SCL | 76.93(2.99) | 39.15(2.84) | 58.04(0.94) |     4.3     |   AudioSet   |

## To cite this work:
```
@INPROCEEDINGS{10248130,
  author={Moummad, Ilyass and Farrugia, Nicolas},
  booktitle={2023 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)}, 
  title={Pretraining Respiratory Sound Representations using Metadata and Contrastive Learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/WASPAA58266.2023.10248130}}
```
