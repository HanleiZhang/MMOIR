for method in 'mult'
do
    for dataset in 'IEMOCAP-DA'
    do
        for text_backbone in 'bert-base-uncased' 
        do    # for video_feats in 'swin-roi'
            for video_feats in 'swin-full'
            do
                for audio_feats in 'wavlm'
                    # for audio_feats in 'wavlm_feats'
                do
                    python run.py \
                    --dataset $dataset \
                    --ood_dataset 'IEMOCAP-DA-OOD' \
                    --data_path '/home/sharing/Datasets' \
                    --logger_name ${method} \
                    --multimodal_method $method \
                    --method ${method} \
                    --data_mode 'multi-class' \
                    --dialogue_mode 'single_turn' \
                    --tune \
                    --train \
                    --save_results \
                    --output_path 'outputs/single_turn' \
                    --gpu_id '1' \
                    --video_feats $video_feats \
                    --audio_feats $audio_feats \
                    --text_backbone $text_backbone \
                    --config_file_name ${method}_IEMOCAP \
                    --results_file_name 'results_iemocap_mult.csv' 
                done
            done    
        done
    done
done