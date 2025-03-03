#!/usr/bin/env python3

import json
import os
import os.path
import sys
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.simplefilter('ignore')


def F1_Recall_Precision(predictions, gold):
    pred_labels = []
    gold_labels = []
    for key in predictions.keys():
        if predictions[key]["Prediction"] == "Entailment":
            pred_labels.append(1)
        elif predictions[key]["Prediction"] == "Contradiction":
            pred_labels.append(0)
        else:
            pred_labels.append(-1)
    for inst in gold:
        if inst["label"] == "Entailment":
            gold_labels.append(1)
        elif inst["label"] == "Contradiction":
            gold_labels.append(0)
        else:
            gold_labels.append(-1)
            
    F1 = f1_score(gold_labels, pred_labels, average="macro")
    Recall = precision_score(gold_labels, pred_labels, average="macro")
    Precision = recall_score(gold_labels, pred_labels, average="macro")
    return F1, Recall, Precision


def main():

    # Load files
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    pred_dir = os.path.join(input_dir, 'res')
    gold_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(pred_dir):
        raise RuntimeError('{} does not exist'.format(pred_dir))

    if not os.path.isdir(gold_dir):
        raise RuntimeError('{} does not exist'.format(gold_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gold_filename = os.path.join(gold_dir, 'gold_pol_final.json')
    pred_filename = os.path.join(pred_dir, 'results.json')

    with open(pred_filename) as json_file:
        predictions = json.load(json_file)

    with open(gold_filename) as json_file:
        gold = json.load(json_file)


    # Test Set F1, Recall, Precision
    Control_F1, Control_Rec, Control_Prec = F1_Recall_Precision(predictions, gold)


    output_filename = os.path.join(output_dir, 'scores.txt')
    with open(output_filename, 'w') as f:
        print('Control_F1: ', Control_F1, file=f)
        print('Control_Recall: ', Control_Rec, file=f)
        print('Control_Precision: ', Control_Prec, file=f)
        
if '__main__' == __name__:
    main()









