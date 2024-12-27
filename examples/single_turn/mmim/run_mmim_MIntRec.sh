for method in 'mmim'
do
    for dataset in 'MIntRec'
    do
        for text_backbone in 'bert-base-uncased' 
        do
            for ood_detection_method in 'ma'
            do  
                for video_feats in 'swin-full'
                do
                    for audio_feats in 'wavlm'
                    do
                        python run.py \
                        --dataset $dataset \
                        --ood_dataset 'MIntRec-OOD' \
                        --data_path '/home/sharing/Datasets' \
                        --logger_name ${method}_${ood_detection_method} \
                        --multimodal_method $method \
                        --method ${method}\
                        --dialogue_mode 'single_turn' \
                        --train \
                        --test_ood \
                        --test_mode 'ood_det' \
                        --tune \
                        --save_results \
                        --save_model \
                        --output_path '/home/sharing/disk1/zhoushihao/haigezhu/MMOIR/single_turn' \
                        --gpu_id '2' \
                        --video_feats $video_feats \
                        --audio_feats $audio_feats \
                        --text_backbone $text_backbone \
                        --config_file_name ${method}_MIntRec \
                        --results_file_name 'results_mintrec_mmim_1.csv' \
                        #--config_file_name ${method}_{$dataset} \
                    done
                done
            done
        done
    done
done