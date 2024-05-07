#!/usr/bin/env bash

SEED1=$1
SEED2=$2

for classifier_name in resnetEcg; do
  for t in 50; do
    for adv_eps in 0.062745098; do
      for seed in $SEED1; do
        for data_seed in $SEED2; do

          CUDA_VISIBLE_DEVICES=0,1,2 python3 eval_sde_adv_bpda.py --exp ./exp_results --config tnmg.yml \
            --t $t \
            --adv_eps $adv_eps \
            --adv_batch_size 2 \
            --domain tnmg \
            --classifier_name $classifier_name \
            --seed $seed \
            --data_seed $data_seed \
            --diffusion_type DeScoDECG \

        done
      done
    done
  done
done
