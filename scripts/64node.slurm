#!/bin/bash
#SBATCH -A m4207
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 0:00:50
#SBATCH -N 64
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --mail-user=sli@tacc.utexas.edu
#SBATCH --mail-type=begin,end
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

CMD="./scripts/run_imagenet.sh    \
    --train-dir=/pscratch/sd/s/sli/data/ILSVRC2012_img_train/  \
    --val-dir=/pscratch/sd/s/sli/data/ILSVRC2012_img_val/ 
    --batch-size=128 " #--base-lr=0.011328
RMCMD="rm /pscratch/sd/s/sli/code/kfac_pytorch/logs/torch_imagenet/checkpoint*"

# Lars
# trust_coef_list=(0.007 0.008 0.009 0.01 0.011 0.012 0.015 )
# for trust_coef in ${trust_coef_list[@]}
# do
CURCMD="$CMD --lars --trust-coef=0.001 \
--kfac-inv-update-steps=0 --epochs=64 \
--warmup-epochs=18 --momentum=0.928 "
echo $CURCMD
$CURCMD   
$RMCMD

#     # CURCMD="$CMD --lars --trust-coef=$trust_coef \
#     # --kfac-inv-update-steps=0 "
#     # echo $CURCMD
#     # $CURCMD  
#     # $RMCMD
# done

#SGD
CURCMD="$CMD --base-lr=0.05 --kfac-inv-update-steps=0 \
--epochs=90 --lr-decay 30 60 80"
echo $CURCMD
$CURCMD   
$RMCMD

# CURCMD="$CMD --base-lr=0.009375 --kfac-inv-update-steps=0 --epochs=90 --lr-decay 30 60 80"
# echo $CURCMD
# $CURCMD   
# $RMCMD

# #LARS
# CURCMD="$CMD --kfac-inv-update-steps=0 --lars --epochs=90 --trust-coef=0.001"
# echo $CURCMD
# $CURCMD   
# $RMCMD

#K-FAC + SGD
# $RMCMD
# CURCMD="$CMD --base-lr=0.05"
# echo $CURCMD
# $CURCMD   
# $RMCMD


# #K-FAC Damping Decay
# CURCMD="$CMD --decay "
# echo $CURCMD
# $CURCMD   
# $RMCMD

# #K-FAC + LARS
# CURCMD="$CMD --lars --trust-coef=0.003 "
# echo $CURCMD
# $CURCMD   
# $RMCMD

# #K-FAC Damping Decay + LARS
# CURCMD="$CMD --decay --lars  --trust-coef=0.003 "
# echo $CURCMD
# $CURCMD   
# $RMCMD
