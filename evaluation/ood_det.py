
import numpy as np
from .score_func import ood_detection_map
from utils.metrics import OOD_Metrics

def ood_detection(args, inputs):
    ood_metrics = OOD_Metrics(args)
    ood_detection_func = ood_detection_map[args.ood_detection_method]
    scores = ood_detection_func(args, inputs)
    binary_labels = np.array([1 if x != args.ood_label_id else 0 for x in inputs['y_true']])
            
    ood_test_scores = ood_metrics(scores, binary_labels, show_results = True)
    return  ood_test_scores
            