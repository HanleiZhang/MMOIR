#!/usr/bin/bash

#1、0版本

# --test_mode 'ood_cls' \
# --test_ood \
for method in 'sdif'
do
    for dataset in 'MIntRec' 
    do
        for text_backbone in 'bert-base-uncased' 
        do
            for ood_detection_method in 'ma'  
            do
                # for video_feats in 'swin-roi'
                for video_feats in 'resnet-50'
                do
                    for audio_feats in 'wav2vec2'
                        # for audio_feats in 'wavlm_feats'
                    do
                        python run.py \
                        --dataset $dataset \
                        --ood_dataset 'MIntRec-OOD' \
                        --data_path '/home/sharing/Datasets' \
                        --logger_name ${method}_${ood_detection_method} \
                        --multimodal_method $method \
                        --method ${method}\
                        --ablation_type 'full' \
                        --dialogue_mode 'single_turn' \
                        --ood_detection_method $ood_detection_method \
                        --train \
                        --test_ood \
                        --test_mode 'ood_det' \
                        --tune \
                        --save_results \
                        --save_model \
                        --output_path '/home/sharing/disk1/zhoushihao/zhou' \
                        --gpu_id '1' \
                        --video_feats $video_feats \
                        --audio_feats $audio_feats \
                        --text_backbone $text_backbone \
                        --config_file_name ${method}_MIntRec \
                        --results_file_name 'results_mintrec_sdif.csv' \

                    done
                done
            done
        done
    done
done
