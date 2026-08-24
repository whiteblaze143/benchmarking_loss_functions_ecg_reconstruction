#!/bin/bash

for milestone in {1..10}; do
    echo "==============================="
    echo "Running synthesis with milestone: $milestone"
    echo "==============================="

    # Step 1: 合成数据
    python /data1/1shared/jinjiarui/run/MLA-Diffusion/main.py --name PTBXL \
        --config_file config/PTBXL/ptbxl_cond.yaml \
        --gpu 5 \
        --condition_type 1 \
        --synthesis_channels 1,2,3,4,5,6,7,8,9,10,11 \
        --output ./results/cond_syn/ \
        --tensorboard \
        --mode synthesis \
        --milestone "$milestone"

    echo "Finished synthesis for milestone: $milestone"

    # Step 2: 评估结果
    echo "Running evaluation for milestone: $milestone"
    python /data1/1shared/jinjiarui/run/MLA-Diffusion/evaluation/compute_metric.py \
        --data_folder results/cond_syn/PTBXL \
        --notes "milestone $milestone"

    echo "Completed milestone: $milestone"
    echo "-----------------------------------"
done
