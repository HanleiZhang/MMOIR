#!/usr/bin/bash

for method in 'tcl_map'
do
    for text_backbone in 'bert-base-uncased'
    do
        for ood_detection_method in  'ma'  
        do
            for video_feats in 'swin-full'
            do
                for audio_feats in 'wavlm'
                do
                    python run.py \
                    --dataset 'MIntRec' \
                    --ood_dataset 'MIntRec-OOD' \
                    --data_path '/home/sharing/Datasets' \
                    --logger_name ${method}_${ood_detection_method} \
                    --multimodal_method $method \
                    --method ${method}\
                    --ablation_type 'full' \
                    --dialogue_mode 'single_turn' \
                    --ood_detection_method $ood_detection_method \
                    --test_ood \
                    --test_mode 'ood_det' \
                    --train \
                    --tune \
                    --save_results \
                    --save_model \
                    --output_path '/home/sharing/single_turn/tcl_map_MIntRec_1' \
                    --gpu_id '1' \
                    --video_feats $video_feats \
                    --audio_feats $audio_feats \
                    --text_backbone $text_backbone \
                    --config_file_name ${method}_MIntRec \
                    --results_file_name 'results_TCL_MAP_MIntRec_719.csv'
                done
            done
        done
    done
done
 