#!/bin/bash
date
dataset_list=("amazon_beauty") # "amazon_beauty","amazon_electronic","yelp"
echo ${dataset_list}
cuda_num_list=(0 0 0)
length=${#dataset_list[@]}
for ((i=0; i<${length}; i++));
do
{
    dataset=${dataset_list[i]}
    cuda_num=${cuda_num_list[i]}
    for model in  sasrec din gru4rec # bert4rec mamba4rec sasrec din gru4rec
    do
    {
          for type in  func_base_train 
          do
            {
              bash ${type}.sh ${dataset} ${model} ${cuda_num}
            } &
          done
    } &
    done
} &
done
wait
date
# bash _0_func_base_train.sh amazon_beauty sasrec 0
# bash _0_func_finetune_train.sh amazon_beauty sasrec 0
# bash _0_func_duet_train.sh amazon_beauty sasrec 0