# PKDG
By Jianan Xia, Lili Zeng, Tengfei Li, Jingyan Xue, Chuanchuan Zhao, Xuezhong Zhou.
This repository contains an official implementation of PKDG

## Environment
This code is developed using on Python 3.9.18 and Pytorch 1.12.1 with NVIDIA GPUs. Training and testing are performed using 1 Intel(R) Xeon(R) Silver 4210 CPU with CUDA 11.4. Other platforms or GPUs are not fully tested.

The required packages of the environment we used to conduct experiments are listed in requirements.txt.
```
pip install -r requirements.txt
```

## Datasets
1. The datasets used in this paper are downloaded from [Fundus](https://github.com/emma-sjwang/Dofe)
2. Dataset architecture as follow:

```
  ├── dataset
     ├── client1
        ├── data_npy
            ├── sample1.npy, sample2.npy, xxxx
        ├── freq_amp_npy
            ├── amp_sample1.npy, amp_sample2.npy, xxxx
     ├── clientxxx
     ├── clientxxx
```

## train
```
python train.py
```

## test
```
python test.py
```

## Acknowledgement
Some of the code is adapted from [ELCFS](https://github.com/liuquande/FedDG-ELCFS?tab=readme-ov-file)


