source ./.bashrc
if [ -f /.dockerenv ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/cv6/frwang/libs/usr_lib64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/intern/zrzhang6/anaconda3/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/intern/zrzhang6/anaconda3/envs/layoutclm
fi

cd ../../runner
# source ~/anaconda3/bin/activate tsr
source ~/anaconda3/bin/activate layoutclm

# export NCCL_DEBUG=info
# export NCCL_SOCKET_IFNAME=eno2.100

export NGPUS=4
export NNODES=1
export cfg=config_1
export work_dir=/work1/cv1/jszhang6/TSR/code/SEMv2/AblationStudy_on_SEMv2/M10/experiments/config_1

if [[ $NNODES -gt 1 ]]; then
    python -m torch.distributed.launch --nproc_per_node $NGPUS --nnodes=$NNODES --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        valid.py --cfg $cfg --work_dir $work_dir
else
	python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=10001 \
        valid.py --cfg $cfg --work_dir $work_dir
fi
