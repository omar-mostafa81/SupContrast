DATA_DIR=./ood_pngs
MEAN="(0.5314865708351135,0.5344920754432678,0.4852450489997864)"
STD="(0.14621567726135254,0.15273576974868774,0.15099382400512695)"

python main_linear.py \
  --dataset path \
  --data_folder ${DATA_DIR} \
  --mean "${MEAN}" \
  --std "${STD}" \
  --batch_size 512 \
  --learning_rate 1 \
  --ckpt save/SupCon/path_models/SimCLR_path_resnet50_lr_0.5_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine/ckpt_epoch_400.pth \
  --num_classes 2 \
  --save_freq 10 \
  --epochs 30 \
