#!/bin/bash

python train.py --dataroot ./data/ImgBlock/carpet --nc 3 --outf ./output --isize 256 --netsize=256 --batchsize 128 --g_lr 0.00002 --d_lr 0.00002 --niter 500 --d_lat_lr 0.00002 --lat_dim 100 --name leather --g_dim 64 --d_dim 64 --print_freq 5
