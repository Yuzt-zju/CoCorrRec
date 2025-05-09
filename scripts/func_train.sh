dataset=$1
model=$2
cuda_num=$3
mode=$4
ITERATION=24300
SNAPSHOT=2430
ARCH_CONF_FILE="configs/${dataset}_conf.json"
GRADIENT_CLIP=5                     # !
BATCH_SIZE=1024

######################################################################
if [ ${dataset} = 'amazon_electronic' ]; then
    LEARNING_RATE=0.005 #0.01 0.005 0.002 0.001
    MAX_EPOCH=10
elif [ ${dataset} = 'yelp' ]; then
    LEARNING_RATE=0.005 #0.01 0.005 0.002 0.001
    MAX_EPOCH=10
elif [ ${dataset} = 'amazon_beauty' ]; then
    LEARNING_RATE=0.005 #0.01 0.005 0.002 0.001
    MAX_EPOCH=10
else
    LEARNING_RATE=0.005
    MAX_EPOCH=10
fi
CHECKPOINT_PATH=../checkpoint/IJCAI2025/${dataset}_${model}/base
######################################################################

echo ${CHECKPOINT_PATH}
echo "Model save to ${CHECKPOINT_PATH}"

dataset_name=${dataset}

USER_DEFINED_ARGS="--model=${model} --num_loading_workers=1 --learning_rate=${LEARNING_RATE} \
--max_gradient_norm=${GRADIENT_CLIP} --max_epoch=${MAX_EPOCH} --batch_size=${BATCH_SIZE} --snapshot=${SNAPSHOT} --max_steps=${ITERATION} --checkpoint_dir=${CHECKPOIN\
T_PATH} --arch_config=${ARCH_CONF_FILE} --data=${dataset_name} --mode=${mode} "

dataset="/home/zhantianyu/data/${dataset^}/ttt4recIR"

train_file="${dataset}/train.txt"
test_file="${dataset}/test.txt"
data="${train_file},${test_file}"

export CUDA_VISIBLE_DEVICES=${cuda_num}
echo ${USER_DEFINED_ARGS}
python ../main/main.py \
--dataset=${data} \
${USER_DEFINED_ARGS}

echo "Training done: ${CHECKPOINT_PATH}"

