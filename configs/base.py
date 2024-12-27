import importlib
from easydict import EasyDict
from .__init__ import pretrained_models_path, video_feats_path, audio_feats_path, feat_dims


class ParamManager:
    
    def __init__(self, args):
        
        args.text_pretrained_model = pretrained_models_path[args.text_backbone]
        args.video_feats_path = video_feats_path[args.video_feats]
        args.audio_feats_path = audio_feats_path[args.audio_feats]
        
        args.text_feat_dim = feat_dims['text'][args.text_backbone]
        args.video_feat_dim = feat_dims['video'][args.video_feats]
        args.audio_feat_dim = feat_dims['audio'][args.audio_feats]
        
        self.args = EasyDict(dict(vars(args)))   
        
def add_config_param(old_args, config_file_name = None):
    
    if config_file_name is None:
        config_file_name = old_args.config_file_name
        
    if config_file_name.endswith('.py'):
        module_name = '.' + config_file_name[:-3]
    else:
        module_name = '.' + config_file_name
    
    method_name = '.' + old_args.method
        
    if old_args.dialogue_mode == 'single_turn':
        dialogue_mode = 'configs.' + old_args.dialogue_mode + method_name
    else:
        dialogue_mode = 'configs.' + old_args.dialogue_mode

    config = importlib.import_module(module_name, dialogue_mode)

    config_param = config.Param
    method_args = config_param(old_args)
    new_args = EasyDict(dict(old_args, **method_args.hyper_param))
    
    return new_args
