
pretrained_models_path = {
    'bert-base-uncased': '/home/sharing/disk1/pretrained_embedding/bert/uncased_L-12_H-768_A-12/',
    'bert-large-uncased':'/home/sharing/disk1/pretrained_embedding/bert/bert-large-uncased',
}

video_feats_path = {
    'swin-roi': 'swin_roi.pkl',#2
    # 'swin-roi': 'swin_roi_binary.pkl',#2
    'resnet-50':'video_feats.pkl',#1
    'swin-full': 'swin_feats.pkl'#tcl  ##IEMOCAP   #MELD-DA
}

audio_feats_path = {
    'wavlm': 'wavlm_feats.pkl',#2  #1  #IEMOCAP   #MELD-DA
    # 'wavlm': 'wavlm_feats_binary.pkl',#2  #1  #IEMOCAP   #MELD-DA
    'wav2vec2':'audio_feats.pkl', #1
}

feat_dims = {
    'text': {
        'bert-base-uncased': 768,
        'bert-large-uncased': 1024
    },
    'video': {
        'swin-roi': 256,
        'swin-full': 1024,
        'resnet-50': 256
    },
    'audio': {
        'wavlm': 768,
        'wav2vec2': 768
    }
}