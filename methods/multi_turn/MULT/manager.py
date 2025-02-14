import torch
import torch.nn.functional as F
import logging
from torch import nn
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from utils.metrics import AverageMeter, Metrics, OOD_Metrics, OID_Metrics
from data.utils import get_dataloader
import numpy as np
from torch import optim
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import itertools
from scipy.stats import norm as dist_model
from utils.mt import generate_context
from itertools import cycle
from utils.functions import softmax_cross_entropy_with_softtarget
from evaluation.oos_cls import doc_classification

__all__ = ['MULT']

class MULT:

    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)
        
        # self.device, self.model = model.device, model.model
        # self.optimizer, self.scheduler = self._set_optimizer(args, self.model)

        self.device, self.model = model.device, model._set_model(args)
        self.optimizer, self.scheduler = self._set_optimizer(args, self.model)

        mm_data = data.data
        mm_dataloader = get_dataloader(args, mm_data)
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            mm_dataloader['train'], mm_dataloader['dev'], mm_dataloader['test']
        

        # self.device, self.model = model.device, model._set_model(args)
        # self.optimizer, self.scheduler = self._set_optimizer(args, self.model)

        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = Metrics(args)
        self.oid_metrics = OID_Metrics(args)
        self.ood_metrics = OOD_Metrics(args)
        #self.ood_detection_func = ood_detection_map[args.ood_detection_method]
        
        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path, self.device)

    def _set_optimizer(self, args, model):
        
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, correct_bias=False)
        
        num_train_optimization_steps = int(args.num_train_examples / args.train_batch_size) * args.num_train_epochs
        num_warmup_steps= int(args.num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
        
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        
        return optimizer, scheduler

    def _train(self, args): 

        early_stopping = EarlyStopping(args)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)
                speaker_ids = batch['speaker_ids'].to(self.device)
                u_mask = batch['umask'].to(self.device)

                text_lengths = torch.sum(text_feats[:, :, 1], dim=2, keepdim=True)  
                audio_lengths = batch['audio_lengths'].to(self.device)
                video_lengths = batch['video_lengths'].to(self.device)

                text_feats = generate_context(args, text_feats, speaker_ids, u_mask, text_lengths, args.context_len)
                audio_feats = generate_context(args, audio_feats, speaker_ids, u_mask, audio_lengths, args.context_len, modality = 'audio')
                video_feats = generate_context(args, video_feats, speaker_ids, u_mask, video_lengths, args.context_len, modality = 'video')

                text_f1, text_f2 = text_feats.shape[-2], text_feats.shape[-1]
                text_feats = text_feats.view(-1, text_f1, text_f2)
                label_ids = label_ids.view(-1)
                u_mask = u_mask.view(-1).bool()
                text_feats = text_feats[u_mask]
                label_ids = label_ids[u_mask]

                audio_f1, audio_f2 = audio_feats.shape[-2], audio_feats.shape[-1]
                audio_feats = audio_feats.view(-1, audio_f1, audio_f2)
                audio_feats = audio_feats[u_mask]

                video_f1, video_f2 = video_feats.shape[-2], video_feats.shape[-1]
                video_feats = video_feats.view(-1, video_f1, video_f2)
                video_feats = video_feats[u_mask]

                is_ids = torch.nonzero(label_ids != args.ood_label_id)
                oos_ids = torch.nonzero(label_ids == args.ood_label_id)

                if len(is_ids) > len(oos_ids):
                    main_e = is_ids
                    cycle_e = cycle(oos_ids)
                else:
                    main_e = oos_ids
                    cycle_e = cycle(is_ids)
                
                batch_size = args.select_bs
                main_e_batches = [main_e[i:i+batch_size] for i in range(0, len(main_e), batch_size)]
                cycle_e_batches = [list(itertools.islice(cycle_e, batch_size)) for _ in range(batch_size)]
                cycle_e_batches = torch.stack([torch.tensor(batch) for batch in cycle_e_batches])

                for step, (m_e, c_e) in enumerate(zip(main_e_batches, cycle_e_batches)):
                    
                    m_select_text_feats = text_feats[m_e].squeeze(1) if text_feats[m_e].ndim == 4 else text_feats[m_e]
                    m_select_video_feats = video_feats[m_e].squeeze(1) if video_feats[m_e].ndim == 4 else video_feats[m_e]
                    m_select_audio_feats = audio_feats[m_e].squeeze(1) if audio_feats[m_e].ndim == 4 else audio_feats[m_e]
                    
                    m_select_label_ids = label_ids[m_e].squeeze(1) if label_ids[m_e].ndim == 2 else label_ids[m_e].unsqueeze(0)

                    if len(c_e) != 0:
                        c_select_text_feats = text_feats[c_e]
                        c_select_video_feats = video_feats[c_e]
                        c_select_audio_feats = audio_feats[c_e]
                        c_select_label_ids = label_ids[c_e]
                    
                    with torch.set_grad_enabled(True):

                        m_logits, last_hiddens = self.model(m_select_text_feats, m_select_video_feats, m_select_audio_feats)

                        if len(c_e) != 0:
                            c_logits, _ = self.model(c_select_text_feats, c_select_video_feats, c_select_audio_feats)

                        
                        if m_select_label_ids[0] != args.ood_label_id:
                            
                            id_loss = self.criterion(m_logits, m_select_label_ids)

                            if len(c_e) != 0:
                                ood_loss = softmax_cross_entropy_with_softtarget(c_logits, args.num_labels, self.device)

                        else:
                            id_loss = self.criterion(c_logits, c_select_label_ids)

                            if len(c_e) != 0:
                                ood_loss = softmax_cross_entropy_with_softtarget(m_logits, args.num_labels, self.device)
                        
                        if len(c_e) != 0:
                            loss = id_loss + args.alpha * ood_loss
                        else:
                            loss = id_loss

                        self.optimizer.zero_grad()
                        
                        loss.backward()
                        loss_record.update(loss.item(), m_select_label_ids.size(0))

                        if args.grad_clip != -1.0:
                            nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                        self.optimizer.step()
                        self.scheduler.step()

            train_outputs = self._get_outputs(args, mode = 'train')   
            eval_outputs = self._get_outputs(args, mode = 'eval')

            inputs = {
                'y_logit_train': train_outputs['y_logit'],
                'y_true_train': train_outputs['y_true'],
                'y_true_test': eval_outputs['y_true'],
                'y_logit_test': eval_outputs['y_logit']
            }
            # eval_y_logit = eval_outputs['y_logit']
            # eval_y_true = eval_outputs['y_true']
            # eval_y_pred = eval_outputs['y_pred']

            # mu_stds = self.cal_mu_std(train_outputs['y_logit'], train_outputs['y_true'], args.num_labels)
            # eval_y_pred = self.classify_doc(args, eval_y_logit, mu_stds)
            # eval_score = self.oid_metrics(eval_y_true, eval_y_pred)['oid_f1']
            eval_score = doc_classification(args, inputs)['oid_f1']

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
                'eval_score': round(eval_score, 4),
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            early_stopping(eval_score, self.model)

            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

        self.best_eval_score = early_stopping.best_score
        self.model = early_stopping.best_model   
        
        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)   

    def batch_iteration(self, args, dataloader):

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        total_features = torch.empty((0, self.model.model.combined_dim)).to(self.device)
       
        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            speaker_ids = batch['speaker_ids'].to(self.device)
            u_mask = batch['umask'].to(self.device)

            text_lengths = torch.sum(text_feats[:, :, 1], dim = 2, keepdim = True)  
            audio_lengths = batch['audio_lengths'].to(self.device)
            video_lengths = batch['video_lengths'].to(self.device)
            
            text_feats = generate_context(args, text_feats, speaker_ids, u_mask, text_lengths)
            audio_feats = generate_context(args, audio_feats, speaker_ids, u_mask, audio_lengths, modality = 'audio')
            video_feats = generate_context(args, video_feats, speaker_ids, u_mask, video_lengths, modality = 'video')

            text_f1, text_f2 = text_feats.shape[-2], text_feats.shape[-1]
            text_feats = text_feats.view(-1, text_f1, text_f2)
            label_ids = label_ids.view(-1)
            u_mask = u_mask.view(-1).bool()

            text_feats = text_feats[u_mask]

            audio_f1, audio_f2 = audio_feats.shape[-2], audio_feats.shape[-1]
            audio_feats = audio_feats.view(-1, audio_f1, audio_f2)
            audio_feats = audio_feats[u_mask]

            video_f1, video_f2 = video_feats.shape[-2], video_feats.shape[-1]
            video_feats = video_feats.view(-1, video_f1, video_f2)
            video_feats = video_feats[u_mask]

            label_ids = label_ids[u_mask]

            select_bs = args.select_bs

            st = 0
            flag = False

            while True:
                
                ed = st + select_bs
                if ed >= u_mask.shape[0]:
                    flag = True
                    ed = u_mask.shape[0]

                select_text_feats = text_feats[st:ed]
                select_video_feats = video_feats[st:ed]
                select_audio_feats = audio_feats[st:ed]
                
                select_label_ids = label_ids[st:ed]
                
                flag_id = torch.any(select_label_ids != args.ood_label_id).item()
                flag_ood = torch.any(select_label_ids == args.ood_label_id).item()

                if flag_id or flag_ood:

                    with torch.set_grad_enabled(False):
                
                        logits, last_hiddens = self.model(select_text_feats, select_video_feats, select_audio_feats)

                        total_logits = torch.cat((total_logits, logits))
                        total_features = torch.cat((total_features, last_hiddens))
                        total_labels = torch.cat((total_labels, select_label_ids))

                st += select_bs
                if flag:
                    break 

        return total_logits, total_labels, total_features 

    def _get_outputs(self, args, mode = 'eval', show_results = False, test_ind = False):

        self.model.eval()
        
        if mode == 'eval':
            total_logits, total_labels, total_features = self.batch_iteration(args, self.eval_dataloader)

        elif mode == 'train':
            total_logits, total_labels, total_features = self.batch_iteration(args, self.train_dataloader)

        elif mode == 'test':
            total_logits, total_labels, total_features = self.batch_iteration(args, self.test_dataloader)
        
        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)
        y_logit = torch.sigmoid(total_logits.detach()).cpu().numpy()

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        y_prob = total_maxprobs.cpu().numpy()
        y_feat = total_features.cpu().numpy()
        
        if test_ind:
            outputs = self.metrics(y_true[y_true != args.ood_label_id], y_pred[y_true != args.ood_label_id])
        else:

            outputs = self.oid_metrics(y_true, y_pred, show_results = show_results)
        
        outputs.update(
            {
                'y_prob': y_prob,
                'y_logit': y_logit,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_feat': y_feat
            }
        )

        return outputs

    def _test(self, args):
        
        test_results = {}
        
        ind_test_results = self._get_outputs(args, mode = 'test', show_results = True, test_ind = True)
        if args.train:
            test_results['best_eval_score'] = round(self.best_eval_score, 4)
        test_results.update(ind_test_results)
        
        if args.test_ood:
            
            ind_train_outputs = self._get_outputs(args, mode = 'train')

            inputs = {
                'y_logit_train': ind_train_outputs['y_logit'],
                'y_true_train': ind_train_outputs['y_true'],
                'y_true_test': ind_test_results['y_true'],
                'y_logit_test': ind_test_results['y_logit']
            }        
            
            oid_test_results = doc_classification(args,inputs)
            test_results.update(oid_test_results)    
        
        return test_results