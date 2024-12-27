# MMOIR

<p align="center">
<a target="_blank">
    <img src="https://img.shields.io/badge/Dataset-Download-blue" alt="Download the dataset">
  </a>
   <a target="_blank">
    <img src="https://img.shields.io/badge/PRs-Welcome-red" alt="PRs are Welcome">
  </a>
</p>
MMOIR is the first multimodal platform for open intent recognition and out-of-distribution detection in conversations. This repo contains convenient methods for adding algorithms and datasets, and we have tested various existing state-of-the-art algorithms in both single-turn and multi-turn conversations.


## Updates 🔥 🔥 🔥 

| Date 	| Announcements 	|
|-	|-	|
| 12/2024  | 🎆 🎆 The first platform for multimodal intent recognition has been released. Refer to the directory [MMOIR](https://github.com/thuiar/MMOIR) for the dataset and codes. |
| 5/2024  | 🎆 🎆  An unsupervised multimodal clustering method (UMC) has been released. Refer to the paper [UMC](https://aclanthology.org/2024.acl-long.2.pdf). |
| 3/2024  | 🎆 🎆 A token-level contrastive learning method with modality-aware prompting (TCL-MAP) has been released. Refer to the paper [TCL-MAP](https://ojs.aaai.org/index.php/AAAI/article/view/29656). |
| 1/2024  | 🎆 🎆 The first large-scale multimodal intent dataset has been released. Refer to the directory [MIntRec2.0](https://github.com/thuiar/MIntRec2.0) for the dataset and codes. Read the paper -- [MIntRec2.0: A Large-scale Benchmark Dataset for Multimodal Intent Recognition and Out-of-scope Detection in Conversations (Published in ICLR 2024)](https://openreview.net/forum?id=nY9nITZQjc).  |
| 10/2022  | 🎆 🎆 The first multimodal intent dataset is published. Refer to the directory [MIntRec](https://github.com/thuiar/MIntRec) for the dataset and codes. Read the paper -- [MIntRec: A New Dataset for Multimodal Intent Recognition (Published in ACM MM 2022)](https://dl.acm.org/doi/abs/10.1145/3503161.3547906).  |
---------------------------------------------------------------------------


## Features

MMOIR has the following features:

- **Large in Scale**: It contains 4 datasets in total, which are MintRec, MintRec2.0, MELD-DA and IEMOCAP. 
  
- **Multi-turn & Multi-party Dialogues**: For example, MintRec2.0 contains 1,245 dialogues with an average of 12 utterances per dialogue in continuous conversations. Each utterance has an intent label in each dialogue. Each dialogue has at least two different speakers with annotated speaker identities for each utterance.

- **Out-of-distribution Detection**: As real-world dialogues are in the open-world scenarios as suggested in [TEXTOIR](https://github.com/thuiar/TEXTOIR), we further include an OOD tag for detecting those utterances that do not belong to any of existing intent classes. They can be used for out-of-distribution detection and improve system robustness.

## Datasets
Here we provide the details of the datasets in MMOIR. You can download the datasets from the following links.
| Datasets | Source |
|----------|--------|
| [MintRec](https://github.com/thuiar/MIntRec) | [Paper](https://dl.acm.org/doi/abs/10.1145/3503161.3547906) |
| [MintRec2.0](https://github.com/thuiar/MIntRec2.0) | [Paper](https://openreview.net/forum?id=nY9nITZQjc) |
| [MELD-DA](https://github.com/sahatulika15/EMOTyDA) | [Paper](https://aclanthology.org/2020.acl-main.402/) |
| [IEMOCAP-DA](https://github.com/sahatulika15/EMOTyDA) | [Paper](https://aclanthology.org/2020.acl-main.402/) |

## Integrated Models
Here we provide the details of the models in MMOIR.
| Model Name | Source | Published |
|------------|--------|-----------|
| [MULT](./examples/multi_turn/run_mult_multiturn.sh) | [Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7195022/) / [Code](https://github.com/yaohungt/Multimodal-Transformer) | ACL 2019 |
| [MAG_BERT](./examples/multi_turn/run_mag_bert_multiturn.sh) | [Paper](https://aclanthology.org/2020.acl-main.214/) / [Code](https://github.com/WasifurRahman/BERT_multimodal_transformer) | ACL 2020 |
| [MCN](./examples/single_turn/mcn/run_mcn.sh) | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Luo_Multi-Task_Collaborative_Network_for_Joint_Referring_Expression_Comprehension_and_Segmentation_CVPR_2020_paper.html) / [Code](https://github.com/luogen1996/MCN) | CVPR 2020 |
| [CC](./examples/single_turn/cc/run_cc.sh) | [Paper](https://yunfan-li.github.io/assets/pdf/Contrastive%20Clustering.pdf) / [Code](https://github.com/Yunfan-Li/Contrastive-Clustering) | AAAI 2021 |
| [MMIM](./examples/single_turn/mmim/run_mmim_MIntRec.sh) | [Paper](https://aclanthology.org/2021.emnlp-main.723/) / [Code](https://github.com/declare-lab/Multimodal-Infomax) | EMNLP 2021 |
| [sccl](./examples/single_turn/sccl/run_sccl.sh) | [Paper](http://proceedings.mlr.press/v70/yang17b/yang17b.pdf) / [Code](https://github.com/xuyxu/Deep-Clustering-Network) | NAACL 2021 |
| [USNID](./examples/single_turn/usnid/run_usnid.sh) | [Paper](https://ieeexplore.ieee.org/document/10349963) / [Code](https://github.com/thuiar/TEXTOIR/tree/main/open_intent_discovery) | IEEE TKDE 2023 |
| [SDIF](./examples/single_turn/sdif/run_sdif_MIntRec.sh) | [Paper](https://ieeexplore.ieee.org/document/10446922) / [Code](https://github.com/joeying1019/sdif-da) | ICASSP 2024 |
| [TCL_MAP](./examples/single_turn/tcl_map/run_tcl_map_MIntRec.sh) | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/29656) / [Code](https://github.com/thuiar/TCL-MAP) | AAAI 2024 |
| [UMC](./examples/single_turn/umc/run_umc.sh) | [Paper](https://aclanthology.org/2024.acl-long.2.pdf) / [Code](https://github.com/thuiar/UMC) | ACL 2024 |


## Results
Please refer to the [results](./results/README.md) for the detailed results of the models in MMOIR.

## Quick start

1. Use anaconda to create Python environment

   ```
   conda create --name MMOIR python=3.9
   conda activate MMOIR
   ```
2. Install PyTorch (Cuda version 11.2)

   ```
   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
   ```
3. Clone the MMOIR repository.

   ```
   git clone git@github.com:thuiar/MMOIR.git
   cd MMOIR
   ```
4. Install related environmental dependencies

   ```
   pip install -r requirements.txt
   ```
5. Run examples (Take mag-bert as an example, more can be seen [here](./examples))

   ```
   sh examples\multi_turn\run_mag_bert_multiturn.sh
   ```

> Notice: You should correctly set the file path address in the .sh file.


## Extensibility
### a. How to add a new dataset?
1. Prepare Data  
Create a new directory to store your dataset. You should provide the train.tsv, dev.tsv, and test.tsv. You should specify the dataset path in the [.sh file](./examples/multi_turn/run_mag_bert_multiturn.sh)。

2. Dataloader Setting  
You need to add the new dataset name to the benchmarks list in [data](./data/__init__.py). You need to define the intent_labels, max_seq_lengths, ood_data, and other information for the new dataset. For example:
```
'MIntRec':{
  'intent_labels': [
  ],
  'max_seq_lengths': {
      'text': 30, 
      'video': 230, 
      'audio': 480, 
  },
  'ood_data':{
      'MIntRec-OOD': {'ood_label': 'UNK'}
  }
```

3. Features data
To prepare features for video and audio, you need to define the feature files in [features_config](./configs/__init__.py) and prepare the corresponding files in data_path/video_data/ and audio_data/.
```
video_feats_path = {
    'swin-roi': 'swin_roi.pkl',#2
    # 'swin-roi': 'swin_roi_binary.pkl',#2
    'resnet-50':'video_feats.pkl',#1
    'swin-full': 'swin_feats.pkl'#tcl  ##IEMOCAP   #MELD-DA
}
```

### b. How to add a new backbone?
1. Provide a new backbone in [backbones](./backbones/SubNets/__init__.py) and create a new model and file. For example:
```
from .FeatureNets import BERTEncoder, RoBERTaEncoder
# from sentence_transformers import SentenceTransformer

text_backbones_map = {
                    'bert-base-uncased': BERTEncoder,
```
2. Configure the new backbone in [configs](./configs/__init__.py). For example:
```
pretrained_models_path = {
    'bert-base-uncased': '/home/sharing/disk1/pretrained_embedding/bert/uncased_L-12_H-768_A-12/',
    'bert-large-uncased':'/home/sharing/disk1/pretrained_embedding/bert/bert-large-uncased',
}
```

### c. How to add a new method?

1. Provide a new backbone in [backbones](./backbones/FusionNets/__init__.py) and create a new model and file. For example:
```
from .mag_bert import MAG_BERT
```
   
2. Configure the parameters for the new method in [configs](./configs/__init__.py), for example [mag_bert_config](./configs/multi_turn/mag_bert_MIntRec2.py).


3. Add the new method in [method](./methods/multi_turn/__init__.py) and create a new model and file, for example [mag_bert](./methods/multi_turn/MAG_BERT/manager.py). You need to define the optimizer, loss function, and methods for training and testing.
```
from .MAG_BERT.manager import MAG_BERT
from .TEXT.manager import TEXT
from .MULT.manager import MULT


method_map = {
    'mag_bert': MAG_BERT,
    'text': TEXT,
    'mult': MULT,

```

4. Add new examples in [examples](./examples), for example [mag_bert](./examples/multi_turn/run_mag_bert_multiturn.sh).

## Citations

If this work is helpful, or you want to use the codes and results in this repo, please cite the following papers:

* [MIntRec2.0: A Large-scale Dataset for Multimodal Intent Recognition and Out-of-scope Detection in Conversations](https://openreview.net/forum?id=nY9nITZQjc)  
* [MIntRec: A New Dataset for Multimodal Intent Recognition](https://dl.acm.org/doi/10.1145/3503161.3547906)
* [Unsupervised Multimodal Clustering for Semantics Discovery in Multimodal Utterances](https://aclanthology.org/2024.acl-long.2.pdf)
* [Token-Level Contrastive Learning with Modality-Aware Prompting for Multimodal Intent Recognition](https://ojs.aaai.org/index.php/AAAI/article/view/29656)

```
@inproceedings{MIntRec2.0,
   title={{MI}ntRec2.0: A Large-scale Benchmark Dataset for Multimodal Intent Recognition and Out-of-scope Detection in Conversations},
   author={Zhang, Hanlei and Wang, Xin and Xu, Hua and Zhou, Qianrui and Su, Jianhua and Zhao, Jinyue and Li, Wenrui and Chen, Yanting and Gao, Kai},
   booktitle={The Twelfth International Conference on Learning Representations},
   year={2024},
   url={https://openreview.net/forum?id=nY9nITZQjc}
}
```
```
@inproceedings{MIntRec,
   author = {Zhang, Hanlei and Xu, Hua and Wang, Xin and Zhou, Qianrui and Zhao, Shaojie and Teng, Jiayan},
   title = {MIntRec: A New Dataset for Multimodal Intent Recognition},
   year = {2022},
   booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
   pages = {1688–1697},
}
```
```
@inproceedings{UMC,
    title = "Unsupervised Multimodal Clustering for Semantics Discovery in Multimodal Utterances",
    author = "Zhang, Hanlei and Xu, Hua and Long, Fei and Wang, Xin and Gao, Kai",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2024",
    url = "https://aclanthology.org/2024.acl-long.2",
    doi = "10.18653/v1/2024.acl-long.2",
    pages = "18--35",
}
```
```
@inproceedings{TCL-MAP,
  title={Token-level contrastive learning with modality-aware prompting for multimodal intent recognition},
  author={Zhou, Qianrui and Xu, Hua and Li, Hao and Zhang, Hanlei and Zhang, Xiaohan and Wang, Yifan and Gao, Kai},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={15},
  pages={17114--17122},
  year={2024}
}
```
