import json
import os
import sys
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, precision_score, recall_score


def random_classifer(gold):

    gold_labels = []

    for inst in gold:
        if inst["label"] == "Entailment":
            gold_labels.append(1)
        elif inst["label"] == "Contradiction":
            gold_labels.append(0)
    gold_label_size = len(gold_labels)

    np.random.seed(seed=42)
    X = np.random.randint(1, size=gold_label_size)
    print(X)
    dummy_clf = DummyClassifier(strategy="uniform")
    dummy_clf.fit(X, gold_labels)
    pred_labels = dummy_clf.predict(X)
    F1 = f1_score(gold_labels, pred_labels, average="macro")
    Recall = precision_score(gold_labels, pred_labels, average="macro")
    Precision = recall_score(gold_labels, pred_labels, average="macro")
    return F1, Recall, Precision

def main():
    
    # Load files
    input_dir = sys.argv[1]
    gold_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(gold_dir):
        raise RuntimeError('{} does not exist'.format(gold_dir))
    
    gold_filename = os.path.join(gold_dir, 'gold_pol_final.json')
    with open(gold_filename) as json_file:
        gold = json.load(json_file)
    
    Control_F1, Control_Rec, Control_Prec = random_classifer(gold)
    print(Control_F1)
    print(Control_Rec)
    print(Control_Prec)

if '__main__' == __name__:
    main()

