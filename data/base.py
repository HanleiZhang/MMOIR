import os
import logging
import csv
import random
import numpy as np
import copy

from .mm_pre import MMDataset,AuGDataset
from .text_pre import get_t_data
from .utils import get_v_a_data
from .text_pre import TextDataset
from .__init__ import benchmarks
from .text_pre import get_ood_text_dataset
from .mm_pre import get_ood_mm_dataset

__all__ = ['DataManager']

class DataManager:
    
    def __init__(self, args):
        
        self.logger = logging.getLogger(args.logger_name)

        bm = benchmarks[args.dataset]
        max_seq_lengths = bm['max_seq_lengths']

        args.text_seq_len = max_seq_lengths['text']
        args.video_seq_len = max_seq_lengths['video']
        args.audio_seq_len = max_seq_lengths['audio']

        # args.data_path = os.path.join(args.data_path, args.dataset)
        if args.clustering:
            self.label_list = copy.deepcopy(bm["intent_labels"])
        else:
            self.label_list = bm["intent_labels"]
        self.logger.info('Lists of intent labels are: %s', str(self.label_list))  
        
        args.ood_label_id = len(self.label_list)
        args.num_labels = len(self.label_list) 

        label_map = {}
        for i, label in enumerate(self.label_list):
            label_map[label] = i

        if args.dataset == 'MIntRec2.0':
            speaker_map = {}
            for i, speaker_name in enumerate(bm['speaker_list']):
                speaker_map[speaker_name] = i

            args.speaker_map = speaker_map
        
        if args.dialogue_mode == 'single_turn':
            if args.clustering:
                self.data, self.train_outputs = get_clu_data(args, bm, label_map, self.logger)
            else:
                self.data = prepare_data(args, self.logger, self.label_list, bm)

        elif args.dialogue_mode == 'multi_turn':
            self.data = prepare_multiturn_data(args, self.logger, self.label_list, bm)

# Multi-turn dialogues
def dialogue_merge(outputs, mode, data_args, elem):

    temp = {}
    temp_utt_id = {}

    for i, (key, v) in enumerate(zip(data_args[mode + '_data_index'], elem)):

        dia_id = key.split('_')[0][3:]
        utt_id = int(key.split('_')[1][3:])

        if dia_id not in temp.keys():
            temp[dia_id] = []
            temp_utt_id[dia_id] = {}

        if utt_id not in temp_utt_id[dia_id].keys():
            temp_utt_id[dia_id][utt_id] = []

        temp_utt_id[dia_id][utt_id].append(v)

    for k in temp_utt_id.keys():
        sorted_temp = []
        for j in sorted (temp_utt_id[k]) : 
            sorted_temp.append(temp_utt_id[k][j][0])
            
        temp[k] = sorted_temp
    
    keys = list(temp.keys())

    new_keys = {keys[i]: i for i in range(len(keys))}

    return new_keys, temp

def singleturn2multiturn(args, outputs, data_args):

    modality_list = []
    speaker_ids_list = []
    label_ids_list = []

    for key in outputs.keys():

        if key in ['text_data', 'video_data', 'audio_data']:
            modality_list.append(key)
        if key.endswith('speaker_ids'):
            speaker_ids_list.append(key)
        if key.endswith('label_ids'):
            label_ids_list.append(key)

    for mode in ['train', 'dev', 'test'] :

        for modality in modality_list: 
        
            if modality == 'text_data':

                feats = outputs[modality][mode]
                keys, infos = dialogue_merge(outputs, mode, data_args, feats)
                results = {keys[k]: v for k, v in infos.items()}
                outputs[modality][mode] = results

            else:
                feats = outputs[modality][mode]['feats']
                keys, infos = dialogue_merge(outputs, mode, data_args, feats)
                results = {keys[k]: v for k, v in infos.items()}
                outputs[modality][mode]['feats'] = results

                lengths = outputs[modality][mode]['lengths']
                keys, infos = dialogue_merge(outputs, mode, data_args, lengths)
                results = {keys[k]: v for k, v in infos.items()}
                outputs[modality][mode]['lengths'] = results

    for speaker_ids_name in speaker_ids_list:

        speaker_ids = outputs[speaker_ids_name]
        keys, infos = dialogue_merge(outputs, speaker_ids_name.split('_')[0], data_args, speaker_ids)
        results = {keys[k]: v for k, v in infos.items()}

        outputs[speaker_ids_name] = results

    for label_ids_name in label_ids_list:

        label_ids = outputs[label_ids_name]
        keys, infos = dialogue_merge(outputs, label_ids_name.split('_')[0], data_args, label_ids)
        results = {keys[k]: v for k, v in infos.items()}      

        outputs[label_ids_name] = results

    return outputs

def prepare_multiturn_data(args, logger, label_list, bm): 

    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    ood_bm = bm['ood_data'][args.ood_dataset]
    label_map[ood_bm['ood_label']] = args.ood_label_id

    total_data_path = os.path.join(args.data_path, args.dataset, 'total')   
    
    total_outputs = get_data(args, logger, total_data_path, ood_bm, label_map)

    train_label_ids, dev_label_ids, test_label_ids = total_outputs['train_label_ids'], total_outputs['dev_label_ids'], total_outputs['test_label_ids']
    train_speaker_ids, dev_speaker_ids, test_speaker_ids = total_outputs['train_speaker_ids'], total_outputs['dev_speaker_ids'], total_outputs['test_speaker_ids']

    if args.method in ['text']:

        text_data = total_outputs['text_data']
       
        text_train_data = TextDataset(train_label_ids, text_data['train'], speaker_ids = train_speaker_ids, multi_turn = True)
        text_dev_data = TextDataset(dev_label_ids, text_data['dev'], speaker_ids = dev_speaker_ids, multi_turn = True)
        text_test_data = TextDataset(test_label_ids, text_data['test'], speaker_ids = test_speaker_ids, multi_turn = True)

        data = {'train': text_train_data, 'dev': text_dev_data, 'test': text_test_data} 

    else:
        
        text_data = total_outputs['text_data']
        video_data = total_outputs['video_data']
        audio_data = total_outputs['audio_data']

        mm_train_data = MMDataset(train_label_ids, text_data['train'], video_data['train'], audio_data['train'], train_speaker_ids, multi_turn = True)
        mm_dev_data = MMDataset(dev_label_ids, text_data['dev'], video_data['dev'], audio_data['dev'], dev_speaker_ids, multi_turn = True)
        mm_test_data = MMDataset(test_label_ids, text_data['test'], video_data['test'], audio_data['test'], test_speaker_ids, multi_turn = True)

        data = {'train': mm_train_data, 'dev': mm_dev_data, 'test': mm_test_data}

    return data

# Single-turn dialogues
def prepare_data(args, logger, label_list, bm):    

    def get_other_hyper(inputs):

        other_hyper = {}
        other_hyper['train'] = {}
        other_hyper['dev'] = {}
        other_hyper['test'] = {}

        for key in inputs.keys():
            if key not in ['text_data', 'video_data', 'audio_data', 'train_label_ids', 'dev_label_ids', 'test_label_ids']:
                if 'train' in inputs[key]:
                    other_hyper['train'][key] = inputs[key]['train']
                if 'dev' in inputs[key]:
                    other_hyper['dev'][key] = inputs[key]['dev']
                if 'test' in inputs[key]:
                    other_hyper['test'][key] = inputs[key]['test']

        return other_hyper
      
    data = {}
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i
    # ood_bm = bm['ood_data'][args.ood_dataset]
    # label_map[ood_bm['ood_label']] = args.ood_label_id

    if args.dataset == 'MIntRec2.0': 
        data_path = os.path.join(args.data_path, args.dataset , 'in-scope')
    else:
        data_path = os.path.join(args.data_path, args.dataset)

    ind_outputs = get_data(args, logger, data_path, bm, label_map)
     
    train_label_ids, dev_label_ids, test_label_ids = ind_outputs['train_label_ids'], ind_outputs['dev_label_ids'], ind_outputs['test_label_ids']
    if args.aug:
        aug_label_ids = ind_outputs['augment_label_ids']
    
    

    if args.method in ['text', 'text_ood']:

        text_data = ind_outputs['text_data']

        text_train_data = TextDataset(train_label_ids, text_data['train'])
        text_dev_data = TextDataset(dev_label_ids, text_data['dev'])
        text_test_data = TextDataset(test_label_ids, text_data['test'])
        
        data = {'train': text_train_data, 'dev': text_dev_data, 'test': text_test_data}

    else:

        text_data, video_data, audio_data = ind_outputs['text_data'], ind_outputs['video_data'], ind_outputs['audio_data']
        ind_other_hyper = get_other_hyper(ind_outputs)

        mm_train_data = MMDataset(train_label_ids, text_data['train'], video_data['train'], audio_data['train'], other_hyper = ind_other_hyper['train'])
        mm_dev_data = MMDataset(dev_label_ids, text_data['dev'], video_data['dev'], audio_data['dev'], other_hyper = ind_other_hyper['dev'])
        mm_test_data = MMDataset(test_label_ids, text_data['test'], video_data['test'], audio_data['test'], other_hyper = ind_other_hyper['test'])


        if args.aug:
            mm_aug_data = AuGDataset(aug_label_ids, text_data['aug'])
        
        data = {'train': mm_train_data, 'dev': mm_dev_data, 'test': mm_test_data}
        
        if args.aug:
            data.update({'aug': mm_aug_data})    

    if args.train_ood or args.test_ood:

        if args.dataset == 'MIntRec2.0': 
            ood_data_path = os.path.join(args.data_path, args.dataset, 'out-of-scope', args.ood_dataset)
        else:
            ood_data_path = os.path.join(args.data_path, args.ood_dataset)

        ood_bm = bm['ood_data'][args.ood_dataset]
        label_map[ood_bm['ood_label']] = args.ood_label_id
        ood_outputs = get_ood_data(args, logger, ood_data_path, bm, label_map)

        if args.method in ['text', 'text_ood']:
            data = get_ood_text_dataset(args, ind_outputs, ood_outputs, data)
                
        else:
            out_other_hyper = get_other_hyper(ood_outputs)
            data = get_ood_mm_dataset(args, ind_outputs, ood_outputs, ind_other_hyper, out_other_hyper, data)

    return data 

# Shared functions
def get_data(args, logger, data_path, bm, label_map):
    
    logger.info('Data preparation...')
    if args.dataset in ['MELD-DA', 'IEMOCAP-DA']:
        text_data_path = os.path.join(data_path, 'non_OOD')
    else:
        text_data_path = data_path
    
    train_outputs = get_indexes_annotations(args, bm, label_map, os.path.join(text_data_path, 'train.tsv'))
    dev_outputs = get_indexes_annotations(args, bm, label_map, os.path.join(text_data_path, 'dev.tsv'))
    test_outputs = get_indexes_annotations(args, bm, label_map, os.path.join(text_data_path, 'test.tsv'))
    if args.method == 'sdif':
        augment_outputs = get_indexes_annotations(args, bm, label_map, os.path.join(text_data_path, 'augment_train.tsv'))

    args.num_train_examples = len(train_outputs['indexes'])
    
    data_args = {
        'data_path': data_path,
        'text_data_path':text_data_path,
        'train_data_index': train_outputs['indexes'],
        'dev_data_index': dev_outputs['indexes'],
        'test_data_index': test_outputs['indexes'],
        'label_map': label_map
    }
    if args.method == 'sdif':
        data_args.update({'augment_data_index': augment_outputs['indexes']})
 
    if args.method in ['text', 'text_ood']:

        text_data = get_t_data(args, data_args)

        outputs = {
            'train_label_ids': train_outputs['label_ids'],
            'dev_label_ids': dev_outputs['label_ids'],
            'test_label_ids': test_outputs['label_ids'],
        }

        outputs['text_data'] = text_data['features']
        text_data.pop('features')
        outputs.update(text_data)

    else:
        text_data = get_t_data(args, data_args)

        video_feats_path = os.path.join(data_args['data_path'], args.video_data_path, args.video_feats_path)
        video_data = get_v_a_data(data_args, video_feats_path, args.video_seq_len)

        audio_feats_path = os.path.join(data_args['data_path'], args.audio_data_path, args.audio_feats_path)
        audio_data = get_v_a_data(data_args, audio_feats_path, args.audio_seq_len)  
        
        outputs = {
            'video_data': video_data,
            'audio_data': audio_data,
            'train_label_ids': train_outputs['label_ids'],
            'dev_label_ids': dev_outputs['label_ids'],
            'test_label_ids': test_outputs['label_ids']
        }

        if args.method == 'sdif':
            outputs.update({'augment_label_ids':augment_outputs['label_ids']})

        outputs['text_data'] = text_data['features']    
        text_data.pop('features')
        outputs.update(text_data)

    
    if args.dialogue_mode == 'multi_turn':
        outputs.update({
            'train_speaker_ids': train_outputs['speaker_ids'],
            'dev_speaker_ids': dev_outputs['speaker_ids'],
            'test_speaker_ids': test_outputs['speaker_ids']
        })
        outputs = singleturn2multiturn(args, outputs, data_args)

    return outputs

def get_ood_data(args, logger, data_path, bm, label_map):
    
    logger.info('OOD Data preparation...')

    train_outputs = {}
    dev_outputs = {}
    test_outputs = {}
    data_args = {
        'data_path': data_path,
        'label_map': label_map
    }

    if args.train_ood:
        
        train_outputs = get_indexes_annotations(args, bm, label_map, os.path.join(data_path, 'train.tsv'))
        dev_outputs = get_indexes_annotations(args, bm, label_map, os.path.join(data_path, 'dev.tsv'))
        args.num_train_examples += len(train_outputs['indexes'])
        data_args.update({'train_data_index': train_outputs['indexes'],'dev_data_index': dev_outputs['indexes']})
    
    if args.test_ood:
        
        test_outputs = get_indexes_annotations(args, bm, label_map, os.path.join(data_path, 'test.tsv'))
        data_args.update({'data_path': data_path, 'test_data_index': test_outputs['indexes'], 'label_map': label_map})
        
    

    if args.method in ['text', 'text_ood']:

        text_data = get_t_data(args, data_args)

        outputs = {}
        if args.train_ood:
            outputs.update({'train_label_ids': train_outputs['label_ids'], 'dev_label_ids': dev_outputs['label_ids']})

        if args.test_ood:
            outputs.update({'test_label_ids': test_outputs['label_ids']})

        outputs['text_data'] = text_data['features']
        text_data.pop('features')
        outputs.update(text_data)
  

    else:
        text_data = get_t_data(args, data_args)

        video_feats_path = os.path.join(data_args['data_path'], args.video_data_path, args.video_feats_path)
        video_data = get_v_a_data(data_args, video_feats_path, args.video_seq_len)

        audio_feats_path = os.path.join(data_args['data_path'], args.audio_data_path, args.audio_feats_path)
        audio_data = get_v_a_data(data_args, audio_feats_path, args.audio_seq_len)  
        
        outputs = {
            'video_data': video_data,
            'audio_data': audio_data,
        }
        if args.train_ood:
            outputs.update({'train_label_ids': train_outputs['label_ids'], 'dev_label_ids': dev_outputs['label_ids']})

        if args.test_ood:
            outputs.update({'test_label_ids': test_outputs['label_ids']})

        outputs['text_data'] = text_data['features']    
        text_data.pop('features')
        outputs.update(text_data)

    return outputs

def get_clu_data(args, bm, label_map, logger):
    
    data_path = os.path.join(args.data_path, args.dataset)
    
    logger.info('data preparation...')
    
    train_outputs = get_indexes_annotations(args, bm, label_map, os.path.join(data_path, 'train.tsv'))
    dev_outputs = get_indexes_annotations(args, bm, label_map, os.path.join(data_path, 'dev.tsv'))

    train_data_index = train_outputs['indexes']
    train_label_ids = train_outputs['label_ids']
    dev_data_index = dev_outputs['indexes']
    dev_label_ids = dev_outputs['label_ids']
    
    train_data_index = train_data_index + dev_data_index
    train_label_ids = train_label_ids + dev_label_ids

    test_outputs = get_indexes_annotations(args, bm, label_map, os.path.join(data_path, 'test.tsv'))

    test_data_index = test_outputs['indexes']
    test_label_ids = test_outputs['label_ids']

    args.num_train_examples = len(train_data_index)
    
    data_args = {
        'data_path': data_path,
        'train_data_index': train_data_index,
        'test_data_index': test_data_index,
    }
        
    text_data = get_t_data(args, data_args)
        
    video_feats_path = os.path.join(data_args['data_path'], args.video_data_path, args.video_feats_path)
    video_data = get_v_a_data(data_args, video_feats_path, args.video_seq_len)
    
    audio_feats_path = os.path.join(data_args['data_path'], args.audio_data_path, args.audio_feats_path)
    audio_data = get_v_a_data(data_args, audio_feats_path, args.audio_seq_len)
    
    mm_train_data = MMDataset(train_label_ids, text_data['train'], video_data['train'], audio_data['train'])
    mm_test_data = MMDataset(test_label_ids, text_data['test'], video_data['test'], audio_data['test'])

    mm_data = {'train': mm_train_data, 'test': mm_test_data}
    
    train_outputs = {
        'text': text_data['train'],
        'video': video_data['train'],
        'audio': audio_data['train'],
        'label_ids': train_label_ids,
    }
    
    return mm_data, train_outputs

def get_indexes_annotations(args, bm, label_map, read_file_path):

    with open(read_file_path, 'r') as f:

        data = csv.reader(f, delimiter="\t")
        indexes = []
        label_ids = []
        speaker_ids = []

        for i, line in enumerate(data):
            if i == 0:
                continue

            if args.dataset in ['MIntRec', 'MIntRec-OOD']:
                index = '_'.join([line[0], line[1], line[2]])

                indexes.append(index)
                label_id = label_map[line[4]]
                
            elif args.dataset in ['MIntRec2.0', 'MIntRec2.0-OOD']:
                index = '_'.join(['dia' + str(line[0]), 'utt' + str(line[1])])
                indexes.append(index)
                speaker_ids.append(args.speaker_map[line[7]])
                label_id = label_map[line[3]]
            
            elif args.dataset in ['MELD-DA', 'MELD-DA-OOD']:
                label_id = label_map[bm['label_maps'][line[3]]]
                index = '_'.join([line[0], line[1]])
                indexes.append(index)
            
            elif args.dataset in ['IEMOCAP-DA', 'IEMOCAP-DA-OOD']:
                label_id = label_map[bm['label_maps'][line[2]]]
                index = line[0]
                indexes.append(index)

            label_ids.append(label_id)
    
    outputs = {
        'indexes': indexes,
        'label_ids': label_ids,
        'speaker_ids': speaker_ids
    }

    return outputs
