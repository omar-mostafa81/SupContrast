#!/usr/bin/env bash

# unset LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:$CONDA_PREFIX/lib

DATA_DIR=./ood_pngs
MEAN="(0.5314865708351135,0.5344920754432678,0.4852450489997864)"
STD="(0.14621567726135254,0.15273576974868774,0.15099382400512695)"

python main_supcon.py \
  --dataset path \
  --data_folder ${DATA_DIR} \
  --mean "${MEAN}" \
  --std "${STD}" \
  --batch_size 256 \
  --learning_rate 0.5 \
  --weight_decay 1e-4 \
  --temp 0.1 \
  --epochs 400 \
  --cosine \
  --syncBN \
  --method SimCLR \
