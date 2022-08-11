PRELOAD="source /etc/profile ; "
PRELOAD+="module load conda/pytorch ; "
PRELOAD+="conda activate torch-1.10;"
PRELOAD+="source switch-cuda.sh;"
PRELOAD+="source switch-cuda.sh 11.3;"
#PRELOAD+="export OMP_NUM_THREADS=8 ; "

# Arguments to the training script are passed as arguments to this script
CMD="examples/torch_cifar10_resnet.py  $@"

# Example: copy imagenet and extract to /tmp on each worker
# ./scripts/copy_and_extract.sh /path/to/imagenet.tar /tmp/imagenet

# Figure out training environment
if [[ -z "${NODEFILE}" ]]; then
    if [[ -n "${SLURM_NODELIST}" ]]; then
        NODEFILE=/tmp/imagenet_slurm_nodelist
        scontrol show hostnames $SLURM_NODELIST > $NODEFILE
    elif [[ -n "${COBALT_NODEFILE}" ]]; then
        NODEFILE=$COBALT_NODEFILE
    fi
fi
if [[ -z "${NODEFILE}" ]]; then
    RANKS=$HOSTNAME
    NNODES=1
else
    MAIN_RANK=$(head -n 1 $NODEFILE)
    RANKS=$(tr '\n' ' ' < $NODEFILE)
    NNODES=$(< $NODEFILE wc -l)
fi

# Torch Distributed Launcher
LAUNCHER="torchrun "
LAUNCHER+="--nnodes=$NNODES --nproc_per_node=1 --max_restarts 0 "
if [[ "$NNODES" -eq 1 ]]; then
    LAUNCHER+="--standalone "
else
    LAUNCHER+="--rdzv_backend=c10d --rdzv_endpoint=$MAIN_RANK "
fi

# Combine preload, launcher, and script+args into full command
FULL_CMD="$PRELOAD $LAUNCHER $CMD"
echo "Training command: $FULL_CMD"

# Launch the pytorch processes on each worker (use ssh for remote nodes)
RANK=0
for NODE in $RANKS; do
    if [[ "$NODE" == "$HOSTNAME" ]]; then
        echo "Launching rank $RANK on local node $NODE"
        eval $FULL_CMD &
    else
        echo "Launching rank $RANK on remote node $NODE"
        ssh $NODE "cd $PWD; $FULL_CMD" &
    fi
    RANK=$((RANK+1))
done

wait
