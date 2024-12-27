from .FeatureNets import BERTEncoder, RoBERTaEncoder
# from sentence_transformers import SentenceTransformer

text_backbones_map = {
                    'bert-base-uncased': BERTEncoder,
                    'bert-large-uncased': BERTEncoder,
                    'roberta-base': RoBERTaEncoder,
                    'roberta-large': RoBERTaEncoder,
                    # 'distilbert-base-nli-stsb-mean-tokens': SentenceTransformer,
                }