
python main.py --name MIMIC_subset \
    --config_file config/mimic_subset_text_cond.yaml \
    --gpu 7 \
    --condition_type 1 \
    --synthesis_channels 1,2,3,4,5,6,7,8,9,10,11 \
    --output ./results/cond_text_syn/ \
    --tensorboard \
    --mode synthesis \
    --milestone 10 \