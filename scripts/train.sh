#!/bin/bash
date
dataset_list=("amazon_electronic") # "amazon_beauty","amazon_electronic","yelp"
echo ${dataset_list}
cuda_num_list=(0)
length=${#dataset_list[@]}
for ((i=0; i<${length}; i++));
do
{
    dataset=${dataset_list[i]}
    cuda_num=${cuda_num_list[i]}
    for model in cocorrrec
    do
    {
      for type in  func_train
      do 
        {
            for mode in base # base  corr
            do
            {
                bash ${type}.sh ${dataset} ${model} ${cuda_num} ${mode}
            } &
            done
        } &
        done
    } &
    done
} &
done