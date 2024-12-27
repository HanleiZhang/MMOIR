#!/usr/bin/bash

 
for method in 'mult'
do
    for dataset in  'MIntRec2.0'
    do
        for text_backbone in  'bert-large-uncased' 
        do
            for video_feats in 'swin-roi'
            do
               for audio_feats in 'wavlm'
                do
                    python run.py \
                    --dataset $dataset \
                    --ood_dataset 'MIntRec2.0-OOD' \
                    --data_path '/home/sharing/Datasets' \
                    --logger_name ${method} \
                    --multimodal_method $method \
                    --method ${method}\
                    --dialogue_mode 'multi_turn' \
                    --train \
                    --tune \
                    --save_results \
                    --save_model \
                    --output_path '/home/sharing/disk1/zhoushihao/haigezhu' \
                    --gpu_id '3' \
                    --video_feats $video_feats \
                    --audio_feats $audio_feats \
                    --text_backbone $text_backbone \
                    --config_file_name ${method}_MIntRec2 \
                    --results_file_name 'results_mult_multiturn_non_ood.csv'
                done    
            done
        done        
    done    
done
