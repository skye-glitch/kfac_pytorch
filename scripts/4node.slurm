#!/bin/bash
#SBATCH -A m4207
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 11:59:59
#SBATCH -N 4
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
    --batch-size=128 --base-lr=0.05"
RMCMD="rm /pscratch/sd/s/sli/code/kfac_pytorch/logs/torch_imagenet/"

optim_list=('sgd'  'adagrad'  'adam'  'amsgrad'  'adabound'  'amsbound')
for optim in ${optim_list[@]}
do
    CURCMD="$CMD --kfac-inv-update-steps=0 --lr-decay 30 60 80 \
    --epochs=90 --optim=$optim"
    CURRM="$RMCMD$optim*"
    echo $CURCMD
    $CURCMD   
    $CURRM

    CURCMD="$CMD --decay --optim=$optim"
    CURRM="$RMCMD$optim*"
    echo $CURCMD
    $CURCMD   
    $CURRM

    CURCMD="$CMD --optim=$optim"
    CURRM="$RMCMD$optim*"
    echo $CURCMD
    $CURCMD   
    $CURRM
done

# CURCMD="$CMD --lars --trust-coef=0.001 \
# --kfac-inv-update-steps=0 --epochs=90"
# echo $CURCMD
# $CURCMD   
# $RMCMD


# trust_coef_list=(0.0001 0.0005 0.001 0.003 0.01 0.03 0.1 0.3 0.5 1)
# for trust_coef in ${trust_coef_list[@]}
# do
#     CURCMD="$CMD --lars --trust-coef=$trust_coef \
#     --kfac-inv-update-steps=0 --epochs=90"
#     echo $CURCMD
#     $CURCMD   
#     $RMCMD

#     CURCMD="$CMD --lars --trust-coef=$trust_coef \
#     --kfac-inv-update-steps=0 "
#     echo $CURCMD
#     $CURCMD  
#     $RMCMD
# done

#SGD
# CURCMD="$CMD --kfac-inv-update-steps=0 --epochs=90 --lr-decay 30 60 80 \
# --checkpoint-format="2kSGD_{epoch}.pth.tar""
# echo $CURCMD
# $CURCMD   
# $RMCMD

# #LARS
# CURCMD="$CMD --kfac-inv-update-steps=0 --lars --epochs=90 --trust-coef=0.001"
# echo $CURCMD
# CURCMD   
# RMCMD

# #K-FAC + SGD
# CURCMD=$CMD
# echo $CURCMD
# CURCMD   
# RMCMD

# #K-FAC Damping Decay
# CURCMD="$CMD --decay "
# echo $CURCMD
# CURCMD   
# RMCMD

# #K-FAC + LARS
# CURCMD="$CMD --lars --trust-coef=0.003 "
# echo $CURCMD
# CURCMD   
# RMCMD

# #K-FAC Damping Decay + LARS
# CURCMD="$CMD --decay --lars  --trust-coef=0.003 "
# echo $CURCMD
# CURCMD   
# RMCMD
