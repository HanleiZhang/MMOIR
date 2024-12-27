for method in 'mult'
do
    for dataset in 'MELD-DA'
    do
        for text_backbone in 'bert-base-uncased' 
        do    # for video_feats in 'swin-roi'
            for video_feats in 'swin-full'
            do
                for audio_feats in 'wavlm'
                    # for audio_feats in 'wavlm_feats'
                do
                    for ood_detection_method in 'ma'
                    do
                        python run.py \
                        --dataset $dataset \
                        --ood_dataset 'MELD-DA-OOD' \
                        --data_path '/home/sharing/Datasets' \
                        --logger_name ${method} \
                        --multimodal_method $method \
                        --method ${method} \
                        --data_mode 'multi-class' \
                        --dialogue_mode 'single_turn' \
                        --test_mode 'ood_det' \
                        --tune \
                        --train \
                        --test_ood \
                        --save_results \
                        --save_model \
                        --output_path '/home/sharing/disk1/zhoushihao/haigezhu/MMOIR/single_turn' \
                        --gpu_id '1' \
                        --video_feats $video_feats \
                        --audio_feats $audio_feats \
                        --text_backbone $text_backbone \
                        --config_file_name ${method}_MELD-DA \
                        --results_file_name 'results_meld-da_mult_new.csv' 
                    done
                done
            done    
        done
    done
done