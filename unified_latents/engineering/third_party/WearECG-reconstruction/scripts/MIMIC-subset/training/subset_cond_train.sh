
python main.py --name MIMIC_subset \
    --config_file config/MIMIC_subset/mimic_subset_cond.yaml \
    --gpu 6 \
    --condition_type 1 \
    --synthesis_channels 1,2,3,4,5,6,7,8,9,10,11 \
    --output ./results/cond_syn/ \
    --tensorboard \
    --mode synthesis \
    --train
