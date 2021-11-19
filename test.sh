#! /bin/bash

WORKING_DIR="/mnt/sfs_turbo/fzx/mt5-prompt/"

MASTER_ADDR=localhost
MASTER_PORT=$(( ( RANDOM % 10000 )  + 10000 ))
NNODES=1
NODE_RANK=0


GPUS_PER_NODE=8
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# OPTIONS_NCCL="NCCL_DEBUG=info"

# Change for multinode config
MP_SIZE=4




CACHE_PATH="/mnt/sfs_turbo/fzx/prompt/cache/"
DATA_PATH="/mnt/sfs_turbo/fzx/prompt/datasets/"
CKPT_PATH="/mnt/sfs_turbo/fzx/prompt/mt5_origin"
TOKENIZER_PATH="/mnt/sfs_turbo/fzx/prompt/spiece.model"
# CONFIG_PATH="${WORKING_DIR}/configs/model/enc_dec_small_config.json"
# CKPT_PATH=None

LR=$1
WD=$2
BATCH_SIZE=$3
GRAD_ACC=$4
seed=$5
tuning_type=$6

SAVE_PATH="${WORKING_DIR}/results_xxl_pawsx_all_dev_save/test_prefix_${tuning_type}_seed${seed}/t5_finetune_xnli_lr${LR}const_wd${WD}_bs$(($3 * $4 * $GPUS_PER_NODE / $MP_SIZE))_scale100_prompt/"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/configs/deepspeed/ds_tnews.json"
CONFIG_PATH="${WORKING_DIR}/configs/model/enc_dec_xlarge_8_config.json"

if [ "$tuning_type" = "prompt" ]; then
    PROMPT_CONFIG="${WORKING_DIR}/configs/prompt/fp.json"
else
    PROMPT_CONFIG="${WORKING_DIR}/configs/prompt/np.json"
fi



TRAIN_ITER=600000
EPOCHS=10

ENC_LEN=256
DEC_LEN=2

TRAIN_LANG='en'
EVAL_LANG='en,de,es,fr,ja,ko,zh'
TEST_LANG='en,de,es,fr,ja,ko,zh'

OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --enc-seq-length ${ENC_LEN}"
OPTS+=" --dec-seq-length ${DEC_LEN}"
OPTS+=" --train-iters ${TRAIN_ITER}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-file ${LOG_FILE}"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --data-name pawsx"
OPTS+=" --train-lang ${TRAIN_LANG}"
OPTS+=" --eval-lang ${EVAL_LANG}"
OPTS+=" --test-lang ${TEST_LANG}"
OPTS+=" --wd ${WD}"
OPTS+=" --data-impl mmap"
OPTS+=" --lazy-loader"
OPTS+=" --tokenizer-type GPT2BPETokenizer"
OPTS+=" --split 949,50,1"
OPTS+=" --distributed-backend nccl"
OPTS+=" --lr ${LR}"
OPTS+=" --no-load-optim"
OPTS+=" --lr-decay-style linear"
OPTS+=" --clip-grad 1.0"
OPTS+=" --warmup 0.0"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --save-interval 10000000000"
OPTS+=" --eval-interval 200"
OPTS+=" --eval-iters 4000"
OPTS+=" --log-interval 10"
OPTS+=" --checkpoint-activations"
OPTS+=" --deepspeed-activation-checkpointing"
OPTS+=" --fp16"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"
OPTS+=" --do_train"
OPTS+=" --do_valid"
# OPTS+=" --do_eval"
OPTS+=" --eval-batch-size 200"
OPTS+=" --prompt_tune"
OPTS+=" --prompt_config ${PROMPT_CONFIG}"
# OPTS+=" --do_infer"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --max-save 1"
OPTS+=" --seed ${seed}"
OPTS+=" --tuning_type ${tuning_type}"
# OPTS+=" --prompt-path ${PROMPT_PATH}"

CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${WORKING_DIR}/finetune_t5.py ${OPTS}"


echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log

set +x