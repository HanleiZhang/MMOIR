                    # --save_model \
for method in 'umc'
do
    for dataset in 'MIntRec'
    do
        for text_backbone in 'bert-base-uncased' 
        do    
            for video_feats in 'swin-full'
            do
                for audio_feats in 'wavlm'
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
                    --output_path '/home/sharing/disk1/clustering/umc/' \
                    --gpu_id '1' \
                    --video_feats $video_feats \
                    --audio_feats $audio_feats \
                    --text_backbone $text_backbone \
                    --config_file_name ${method}_MIntRec_1 \
                    --results_file_name 'results_mintrec_umc.csv' 
                done
            done    
        done
    done
done