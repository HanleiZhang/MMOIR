import os
import csv
import sys
import torch
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer, T5Tokenizer    #XCLIPProcessor
from torch.utils.data import Dataset

def get_t_data(args, data_args):
    
    if args.text_backbone.startswith('bert'):
        if args.clustering:
            t_data = get_clu_data(args, data_args)
        else:
            t_data = get_data(args, data_args)
    else:
        raise Exception('Error: inputs are not supported text backbones.')

    return t_data

def get_data(args, data_args):

    processor = DatasetProcessor(args)
    if 'text_data_path' in data_args:
        data_path = data_args['text_data_path']
    else:
        data_path = data_args['data_path']
    outputs = {}

    if 'train_data_index' in data_args:
        
        train_examples = processor.get_examples(data_path, 'train') 
        train_feats = get_backbone_feats(args, train_examples)
        
        dev_examples = processor.get_examples(data_path, 'dev')
        dev_feats = get_backbone_feats(args, dev_examples)

        for key in train_feats.keys():
            tmp_outputs = {}
            tmp_outputs[key] = {
                'train': train_feats[key],
                'dev': dev_feats[key],
            }
            outputs.update(tmp_outputs)
    
    if 'test_data_index' in data_args:
        test_examples = processor.get_examples(data_path, 'test')
        test_feats = get_backbone_feats(args, test_examples)

        for key in test_feats.keys():
            if key in outputs:
                outputs[key].update({'test': test_feats[key]})
            else:
                outputs[key] = {'test': test_feats[key]}

    if 'augment_data_index' in data_args:
        augment_examples = processor.get_examples(data_path, 'aug') 
        augment_feats = get_backbone_feats(args, augment_examples)

        for key in augment_feats.keys():
           if key in outputs:
                outputs[key].update({'aug': augment_feats[key]})
            
    return outputs

def get_clu_data(args, data_args):

    processor = DatasetProcessor(args)
    data_path = data_args['data_path']

    train_examples = processor.get_examples(data_path, 'train') 
    dev_examples = processor.get_examples(data_path, 'dev')

    train_examples = train_examples + dev_examples

    train_feats = get_backbone_feats(args, train_examples)

    
    test_examples = processor.get_examples(data_path, 'test')
    test_feats = get_backbone_feats(args, test_examples)

    
    outputs = {
        'train': train_feats['features'],
        'test': test_feats['features']
    }
        
    return outputs


def get_backbone_feats(args, examples):
    
    if args.text_backbone.startswith('bert'):
        tokenizer = BertTokenizer.from_pretrained(args.text_pretrained_model, do_lower_case=True)   

    if args.text_backbone.startswith(('bert')):

        outputs = convert_examples_to_features(args, examples, tokenizer)   
        features = outputs['features']
        features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in features]

        outputs['features'] = features_list

        if args.method == 'tcl_map':
            cons_text_feats = outputs['cons_text_feats']
            cons_text_feats = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in cons_text_feats]
            outputs['cons_text_feats'] = cons_text_feats

        return outputs
    
def get_ood_text_dataset(args, outputs, ood_outputs, data):

    if args.train_ood:
        ood_text_train_data = TextDataset(ood_outputs['train_label_ids'], ood_outputs['text_data']['train'])
        ood_text_dev_data = TextDataset(ood_outputs['dev_label_ids'], ood_outputs['text_data']['dev'])

        data.update({
        'ood_train': ood_text_train_data,
        'ood_dev': ood_text_dev_data,
        })

    if args.test_ood:
        outputs['text_data']['test'].extend(ood_outputs['text_data']['test'])
        outputs['test_label_ids'].extend(ood_outputs['test_label_ids'])

    ood_text_test_data = TextDataset(outputs['test_label_ids'], outputs['text_data']['test'])
                
    data.update({
        'test': ood_text_test_data
    })

    return data

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class DatasetProcessor(DataProcessor):

    def __init__(self, args):
        super(DatasetProcessor).__init__()
        self.use_label_id = True if args.method in ['tcl_map'] else False

        if args.dataset in ['MIntRec']:
            self.select_id = 3
            self.label_id = 4
        elif args.dataset in ['MIntRec2.0']:
            self.select_id = 2
        elif args.dataset in ['MELD-DA']:
            self.select_id = 2
            self.label_id = 3
        elif args.dataset in ['IEMOCAP-DA']:
            self.select_id = 1
            self.label_id = 2
        
    def get_examples(self, data_dir, mode):
        
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'dev':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
        elif mode == 'all':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "all.tsv")), "all")
        elif mode == 'aug':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "augment_train.tsv")), "aug")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        
        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, i)
            text_a = line[self.select_id]
            
            label = line[self.label_id] if self.use_label_id else None

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

def convert_examples_to_features(args, examples, tokenizer):

    max_seq_length = args.text_seq_len
        
    outputs = {}

    if args.method == 'tcl_map':
        
        label_maps = {
                    'g': 'Greeting', 'q': 'Question', 'ans': 'Answer', 'o': 'Statement Opinion', 's': 'Statement Non Opinion', 
                    'ap': 'Apology', 'c': 'Command', 'ag': 'Agreement', 'dag': 'Disagreement', 
                    'a': 'Acknowledge', 'b': 'Backchannel', 'oth': 'Others'
        }
        label_len = args.label_len
        features = []
        cons_features = []
        condition_idx = []
        prefix = ['MASK'] * 3

        max_cons_seq_length = max_seq_length + len(prefix) + label_len
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)
            if args.dataset in ['MIntRec']:
                condition = tokenizer.tokenize(example.label)
            elif args.dataset in ['MELD-DA', 'IEMOCAP-DA']:
                condition = tokenizer.tokenize(label_maps[example.label])

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            # construct augmented sample pair
            cons_tokens = ["[CLS]"] + tokens_a + prefix + condition + (label_len - len(condition)) * ["MASK"] + ["[SEP]"]
            tokens = ["[CLS]"] + tokens_a + prefix + label_len * ["[MASK]"] + ["[SEP]"]

            segment_ids = [0] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            cons_inputs_ids = tokenizer.convert_tokens_to_ids(cons_tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_cons_seq_length - len(input_ids))
            input_ids += padding
            cons_inputs_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_cons_seq_length
            assert len(cons_inputs_ids) == max_cons_seq_length
            assert len(input_mask) == max_cons_seq_length
            assert len(segment_ids) == max_cons_seq_length
            # record the position of prompt
            condition_idx.append(1 + len(tokens_a) + len(prefix))


            features.append(
                InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids)
                            )
            
            cons_features.append(
                InputFeatures(input_ids=cons_inputs_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids)
                            )
        
        args.max_cons_seq_length = max_cons_seq_length
        outputs = {
            'features': features,
            'cons_text_feats': cons_features,
            'condition_idx': condition_idx
        }

    else:   
        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            # if ex_index < 5:
            #     logger.info("*** Example ***")
            #     logger.info("guid: %s" % (example.guid))
            #     logger.info("tokens: %s" % " ".join(
            #         [str(x) for x in tokens]))
            #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #     logger.info(
            #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

            features.append(
                InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids)
                            )
        
        outputs = {
            'features': features
        }
    return outputs

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()

class TextDataset(Dataset):
    
    def __init__(self, label_ids, text_feats, speaker_ids = None, multi_turn = False):
        
        self.label_ids = label_ids
        self.text_feats = text_feats
        self.size = len(self.text_feats)

        self.speaker_ids = speaker_ids
        self.multi_turn = multi_turn

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {
            'text_feats': torch.tensor(self.text_feats[index]),
            'label_ids': torch.tensor(self.label_ids[index]), 
        } 
        
        if self.multi_turn:
            sample.update({
                'speaker_ids': torch.tensor(self.speaker_ids[index]),
                'umask': torch.tensor([1] * len(self.label_ids[index]))
            })
            
        return sample