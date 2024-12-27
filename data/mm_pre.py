from torch.utils.data import Dataset
import torch
import numpy as np

__all__ = ['MMDataset']

class MMDataset(Dataset):
        
    def __init__(self, label_ids, text_data, video_data, audio_data, speaker_ids = None, multi_turn = False, other_hyper = None):
        
        self.label_ids = label_ids
        self.text_data = text_data
        self.video_data = video_data
        self.audio_data = audio_data
        self.size = len(self.text_data)
        self.speaker_ids = speaker_ids
        self.multi_turn = multi_turn

        self.other_hyper = other_hyper
        '''
        Note that the parameters in other_hypers cannot be the same as in the parent class
        '''

        if self.other_hyper is not None:
            for key in other_hyper.keys():
                setattr(self, key, other_hyper[key])  

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        
        sample = {
            'label_ids': torch.tensor(self.label_ids[index]), 
            'text_feats': torch.tensor(self.text_data[index]),
            'video_feats': torch.tensor(np.array(self.video_data['feats'][index])),
            'video_lengths': torch.tensor(np.array(self.video_data['lengths'][index])),
            'audio_feats': torch.tensor(np.array(self.audio_data['feats'][index])),
            'audio_lengths': torch.tensor(np.array(self.audio_data['lengths'][index]))
        } 

        if self.other_hyper is not None:
            
            for key in self.other_hyper.keys():
                sample[key] = torch.tensor(getattr(self, key)[index])


        if self.multi_turn:
            sample.update({
                'speaker_ids': torch.tensor(np.array(self.speaker_ids[index])),
                'umask': torch.tensor(np.array([1] * len(self.label_ids[index])))
            })

        return sample
    
def get_ood_mm_dataset(args, outputs, ood_outputs, ind_other_hyper, out_other_hyper, data):

    if args.train_ood:
        ood_mm_train_data = MMDataset(ood_outputs['train_label_ids'], ood_outputs['text_data']['train'], ood_outputs['video_data']['train'], \
                                      ood_outputs['audio_data']['train'], other_hyper = out_other_hyper['train'])
        ood_mm_dev_data = MMDataset(ood_outputs['dev_label_ids'], ood_outputs['text_data']['dev'], ood_outputs['video_data']['dev'], \
                                    ood_outputs['audio_data']['dev'], other_hyper = out_other_hyper['dev'])

        data.update({
            'ood_train': ood_mm_train_data,
            'ood_dev': ood_mm_dev_data
        })

    if args.test_ood:
        
        outputs['text_data']['test'].extend(ood_outputs['text_data']['test'])
        outputs['test_label_ids'].extend(ood_outputs['test_label_ids'])

        outputs['video_data']['test']['feats'].extend(ood_outputs['video_data']['test']['feats'])
        outputs['video_data']['test']['lengths'].extend(ood_outputs['video_data']['test']['lengths'])

        outputs['audio_data']['test']['feats'].extend(ood_outputs['audio_data']['test']['feats'])
        outputs['audio_data']['test']['lengths'].extend(ood_outputs['audio_data']['test']['lengths'])

        for key in ind_other_hyper['test'].keys():
            ind_other_hyper['test'][key].extend(out_other_hyper['test'][key])
       
        mm_test_data = MMDataset(outputs['test_label_ids'], outputs['text_data']['test'], outputs['video_data']['test'], outputs['audio_data']['test'], other_hyper = ind_other_hyper['test'])

        data.update({
            'test': mm_test_data
        })

    return data

class AuGDataset(Dataset):
        
    def __init__(self, label_ids, text_feats):
        
        self.label_ids = torch.tensor(label_ids)
        self.text_feats = torch.tensor(text_feats)
        self.size = len(self.text_feats)

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {
            'label_ids': self.label_ids[index], 
            'text_feats': self.text_feats[index],
        } 
        return sample



    # #形参可能要使用ood_outputs,ind_outputs
    # if args.test_ood:
    #     # text = ind_outputs['text_data']
    #     # video = ind_outputs['video_data']
    #     # audio = ind_outputs['audio_data']
    #     # train_label_ids = ind_outputs['train_label_ids']
    #     # dev_label_ids = ind_outputs['dev_label_ids']

    #     ind_mm_train_data = MMDataset(ind_outputs['train_label_ids'], ind_outputs['text_data']['train'], ind_outputs['video_data']['train'], \
    #                                   ind_outputs['audio_data']['train'], other_hyper = other_hyper['train'])
    #     ind_mm_dev_data = MMDataset(ind_outputs['dev_label_ids']['dev'], ind_outputs['text_data']['dev'], ind_outputs['video_data']['dev'], \
    #                                 ind_outputs['audio_data']['dev'], other_hyper = other_hyper['dev'])



    # if args.train_ood:
    #     # ood_text = ood_outputs['text_data']
    #     # ood_video = ood_outputs['video_data']
    #     # ood_audio = ood_outputs['audio_data']
    #     # train_label_ids = ood_outputs['train_label_ids']
    #     # dev_label_ids = ood_outputs['dev_label_ids']

    #     ood_mm_train_data = MMDataset(ood_outputs['train_label_ids'], ood_outputs['text_data']['train'], ood_outputs['video_data']['train'], \
    #                                   ood_outputs['audio_data']['train'], other_hyper = other_hyper['train'])
    #     ood_mm_dev_data = MMDataset(ood_outputs['dev_label_ids'], ood_outputs['text_data']['dev'], ood_outputs['video_data']['dev'], \
    #                                 ood_outputs['audio_data']['dev'], other_hyper = other_hyper['dev'])
    


    # if args.test_ood:
    #     ind_outputs['text_data']['test'].extend(ood_outputs['text_data']['test'])
    #     test_label_ids = ind_outputs['test_label_ids'].extend(ood_outputs['ood_test_label_ids'])

    #     ind_outputs['video_data']['test']['feats'].extend(ood_outputs['video_data']['test']['feats'])
    #     ind_outputs['video_data']['test']['lengths'].extend(ood_outputs['video_data']['test']['lengths'])

    #     ind_outputs['audio_data']['test']['feats'].extend(ood_outputs['audio_data']['test']['feats'])
    #     ind_outputs['audio_data']['test']['lengths'].extend(ood_outputs['audio_data']['test']['lengths'])

    #     mm_test_data = MMDataset(test_label_ids, ind_outputs['text_data']['test'], ind_outputs['video_data']['test'], ind_outputs['audio_data']['test'], other_hyper = other_hyper['test'])

    # data.update({
    #     'ood_train': ood_mm_train_data,
    #     'ood_dev': ood_mm_dev_data,
    #     'test': mm_test_data
    # })
    