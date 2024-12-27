#!/usr/bin/bash

for method in 'mult'
do
    for text_backbone in  'bert-large-uncased'  
    do
        for ood_detection_method in 'ma'  
        do
            for video_feats in 'swin-roi'
            do
                for audio_feats in 'wavlm'
                do
                    python run.py \
                    --dataset 'MIntRec2.0' \
                    --ood_dataset 'MIntRec2.0-OOD' \
                    --data_path '/home/sharing/Datasets' \
                    --logger_name ${method}_${ood_detection_method} \
                    --multimodal_method $method \
                    --method ${method}\
                    --dialogue_mode 'single_turn' \
                    --ood_detection_method $ood_detection_method \
                    --train \
                    --tune \
                    --save_results \
                    --save_model \
                    --gpu_id '0' \
                    --video_feats $video_feats \
                    --audio_feats $audio_feats \
                    --text_backbone $text_backbone \
                    --output_path '/home/sharing/disk1/zhoushihao/haigezhu/MMOIR' \
                    --config_file_name ${method}_MIntRec2 \
                    --results_file_name 'results_mintrec2_mult_test.csv'
                done
            done
        done
    done
done

