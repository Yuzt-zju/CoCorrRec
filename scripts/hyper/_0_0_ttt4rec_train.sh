#!/bin/bash
date
# dataset_list=("amazon_beauty" "amazon_cds" "amazon_electronic")
dataset_list=("amazon_beauty")
echo ${dataset_list}
line_num_list=(7828 21189 30819)
cuda_num_list=(3 3 3)
echo ${line_num_list}
length=${#dataset_list[@]}
for ((i=0; i<${length}; i++));
do
{
    dataset=${dataset_list[i]}
    cuda_num=${cuda_num_list[i]}
    for model in ttt4rec
    do
    {
        for init_range in 0.005 0.01 0.02 0.04
        do
        {
          for mini_batch_size in 5 10
          do
          {
            for type in  _0_ttt_train
            do
              {
                bash ${type}.sh ${dataset} ${model} ${cuda_num} ${init_range} ${mini_batch_size}
              } &
            done
          } &
          done
        } &
        done
    } &
    done
} &
done
wait # 等待所有任务结束
date
# bash _0_func_base_train.sh amazon_beauty sasrec 0
# bash _0_func_finetune_train.sh amazon_beauty sasrec 0
# bash _0_func_duet_train.sh amazon_beauty sasrec 0