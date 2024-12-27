for method in 'usnid'
do
    for dataset in 'MIntRec'
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
                    --save_model \
                    --output_path '/home/sharing/usnidmia' \
                    --gpu_id '1' \
                    --video_feats $video_feats \
                    --audio_feats $audio_feats \
                    --text_backbone $text_backbone \
                    --config_file_name ${method}_MIntRec \
                    --results_file_name '22222222222results_mintrec_usnid.csv' 
                done
            done    
        done
    done
done
