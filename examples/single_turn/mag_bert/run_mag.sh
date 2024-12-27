for method in 'mag_bert'
do
    for dataset in 'MIntRec'
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
                        --dataset $dataset \
                        --ood_dataset 'MIntRec-OOD' \
                        --data_path '/home/sharing/Datasets' \
                        --logger_name ${method} \
                        --multimodal_method $method \
                        --method ${method} \
                        --ood_detection_method $ood_detection_method \
                        --data_mode 'multi-class' \
                        --dialogue_mode 'single_turn' \
                        --tune \
                        --train \
                        --test_ood \
                        --test_mode 'ood_det' \
                        --save_results \
                        --output_path '/home/sharing/disk1/zhoushihao/wangpeiwu/single_turn/MIntRec/mag_bert_no_ood' \
                        --gpu_id '0' \
                        --video_feats $video_feats \
                        --audio_feats $audio_feats \
                        --text_backbone $text_backbone \
                        --config_file_name ${method}_MIntRec_TMM \
                        --results_file_name 'results_mintrec_mag_bert_TMM.csv' 
                    done
                done
            done    
        done
    done
done
