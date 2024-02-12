#!/bin/scripts

python Train_event_vid.py   --arch 'PSTA_img_event_cat'\
                  --config_file "./configs/softmax_triplet_prid_event.yml"\
                  --dataset 'prid_event_vid'\
                  --test_sampler 'Begin_interval'\
                  --triplet_distance 'cosine'\
                  --test_distance 'cosine'\
                  --seq_len 4 \
