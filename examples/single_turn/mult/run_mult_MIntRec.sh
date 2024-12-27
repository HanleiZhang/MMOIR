#!/usr/bin/bash

#1、0版本

for method in 'mult'
do
    for dataset in 'MIntRec' 
    do
        for text_backbone in 'bert-base-uncased' 
        do
            for ood_detection_method in 'ma'  
            do
                for video_feats in 'resnet-50'
                do
                    for audio_feats in 'wav2vec2'
                    do
                        python run.py \
                        --dataset 'MIntRec' \
                        --data_path '/home/sharing/Datasets' \
                        --logger_name ${method}_${ood_detection_method} \
                        --multimodal_method $method \
                        --method ${method}\
                        --dialogue_mode 'single_turn' \
                        --train \
                        --tune \
                        --save_results \
                        --gpu_id '3' \
                        --video_feats $video_feats \
                        --audio_feats $audio_feats \
                        --text_backbone $text_backbone \
                        --output_path "/home/sharing/disk1/zhoushihao/wangpeiwu/single_turn/MIntRec/${method}" \
                        --config_file_name ${method}_MIntRec \
                        --results_file_name 'results_mintrec_mult.csv'
                    done
                done
            done
        done
    done
done
