#!/usr/bin/bash

for method in 'cc'
do
    for dataset in 'MIntRec'
    do
        for text_backbone in 'bert-base-uncased' 
        do    # for video_feats in 'swin-roi'
            for video_feats in 'resnet-50'
            do
                for audio_feats in 'wav2vec2'
                    # for audio_feats in 'wavlm_feats'
                do
                    python run.py \
                    --dataset $dataset \
                    --ood_dataset 'MIntRec-OOD' \
                    --data_path '/home/sharing/Datasets' \
                    --logger_name ${method} \
                    --multimodal_method $method \
                    --method ${method} \
                    --data_mode 'multi-class' \
                    --dialogue_mode 'single_turn' \
                    --clustering \
                    --tune \
                    --train \
                    --save_results \
                    --output_path '/home/sharing/disk1/zhoushihao/zhou/clustering' \
                    --gpu_id '3' \
                    --video_feats $video_feats \
                    --audio_feats $audio_feats \
                    --text_backbone $text_backbone \
                    --config_file_name ${method}_MIntRec \
                    --results_file_name 'results_mintrec_cc.csv' 
                done
            done    
        done
    done
done