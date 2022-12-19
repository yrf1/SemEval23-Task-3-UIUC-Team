import os
import json
import torch
import random
import numpy as np
import pandas as pd
from datetime import datetime
from nltk import sent_tokenize
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer


data_dir = "data"
lang_dir = "en"
task_dir_train = "train-articles-subtask-1"
task_label_fname = "train-labels-subtask-1.txt"
labels = pd.read_csv(data_dir+"/"+lang_dir+"/"+task_label_fname, sep="\t", header=None)

data_all_opinion = labels[labels[1]=="opinion"]
data_all_reporting = labels[labels[1]=="reporting"]
data_all_satire = labels[labels[1]=="satire"]

print(data_all_opinion.shape, data_all_reporting.shape, data_all_satire.shape)

from detoxify import Detoxify
from pysentimiento import create_analyzer
analyzer = create_analyzer(task="sentiment", lang="en")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("lighteternal/fact-or-opinion-xlmr-el")

model = AutoModelForSequenceClassification.from_pretrained("lighteternal/fact-or-opinion-xlmr-el")
detox_model = Detoxify('original')

from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

def compute_sentiment(s):
    return analyzer.predict(s).probas

def compute_opiniation(s):
    tok = tokenizer.encode(s, return_tensors="pt", truncation=True)
    return model(tok).logits[0].tolist()

def compute_toxicity(s):
    return detox_model.predict(s)

def compute_bart_MNLI(s):
    return classifier(s, ['opinion','reporting','satire'])["scores"]

feats_tracker = {}
feats_tracker_save_fpath = "t1_feats_tracker_dev.json"
if os.path.exists(feats_tracker_save_fpath):
    with open(feats_tracker_save_fpath, "r") as f:
        feats_tracker = json.load(f)
#for data_by_class in [data_all_opinion, data_all_reporting, data_all_satire]:
#    for idx, data in data_by_class.iterrows():
#        fname, label = data
task_dir_train = "dev-articles-subtask-1"
for fname in os.listdir(data_dir+"/"+lang_dir+"/"+task_dir_train):
    #mean_tracker = []
    if True:
        fname = fname.replace("article","").replace(".txt","")
        with open(data_dir+"/"+lang_dir+"/"+task_dir_train+"/article"+str(fname)+".txt", "r") as f:
            doc_txt = f.read()
        sentiment = compute_sentiment(doc_txt)
        toxicity = compute_toxicity(doc_txt)
        opinion = compute_opiniation(doc_txt)
        #feats = compute_bart_MNLI(doc_txt)
        #feats_tracker[str(fname)].append(feats)
        feats = [sentiment["NEG"],sentiment["NEU"],sentiment["POS"]]
        feats.extend([toxicity["toxicity"],toxicity["insult"],toxicity["identity_attack"]])
        feats.extend(opinion)
        feat_by_sent_tracker = []
        for sent in sent_tokenize(doc_txt):
            sentiment = compute_sentiment(sent)
            toxicity = compute_toxicity(sent)
            opinion = compute_opiniation(sent)
            this_feat_by_sent = [sentiment["NEG"],sentiment["NEU"],sentiment["POS"]]
            this_feat_by_sent.extend([toxicity["toxicity"],toxicity["insult"],toxicity["identity_attack"]])
            this_feat_by_sent.extend(opinion)
            feat_by_sent_tracker.append(this_feat_by_sent)
        mean_by_sent = np.mean(np.array(feat_by_sent_tracker), axis=0).tolist()
        max_by_sent = np.max(np.array(feat_by_sent_tracker), axis=0).tolist()
        feats.extend(mean_by_sent)
        feats.extend(max_by_sent)
        feats.extend(compute_bart_MNLI(doc_txt))
        feats = [float(x) for x in feats]
        feats_tracker[fname] = feats
        #mean_tracker.append(feats)
        with open(feats_tracker_save_fpath, "w") as f:
            json.dump(feats_tracker, f)
    #print(np.mean(np.array(mean_tracker), axis=0), label)

