import os
import copy
import json
import torch
import random
import pandas as pd
from datetime import datetime
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import MarianMTModel, MarianTokenizer

"""
given a .template file in English and a model checkpoint, return the generated test set results 
"""

task_label_fname_test = "test-labels-subtask-3.txt"
test_data_path = "data/po/po_to_en_dev.template"

checkpoint_path = "ckpts/0/en_def_googletrans.pt"

## Load Data
class MyDataset(Dataset):
    def __init__(self, mode="test", cross_val_split_idx=-1, all_labels=None):
        self.data_all, self.data_lang = [], []

        task_label_fname = task_label_fname_test
        
        self.all_labels = [] if mode not in ["val", "dev", "test"] else all_labels
        for lang_dir in ["en", "it","ge","fr","po","ru"]: #os.listdir(data_dir):
            labels = pd.read_csv(data_dir+"/"+lang_dir+"/"+task_label_fname, sep="\t", header=None) \
                     if mode in ["train","pretrain","val"] else None
            with open(test_data_path, "r") as f:
                seg_data = f.read()
                seg_data = [x.split("\t") for x in seg_data.split("\n")[:-1]]
            seg_map = {}
            for d in seg_data:
                if d[0] not in seg_map:
                    seg_map[d[0]] = {}
                seg_map[d[0]][d[1]] = d[-1]
                if mode in ["val","dev", "test"] and lang_dir==lang:
                    for lbl in LABELS_OF_INTEREST:
                        lbls_GT_here = []
                        if mode=="val":        
                            lbls_GT_here = labels[(labels[0]==int(d[0])) & (labels[1]==int(d[1]))][2].values[0]
                            lbls_GT_here = [] if lbls_GT_here!=lbls_GT_here else lbls_GT_here.split(",")
                        self.data_lang.append((d[0],d[1],d[2],lbl,lbls_GT_here, lang_dir))
            if mode in ["pretrain","train"]:
                for idx, data in labels.iterrows():
                    fname, segID, label = data
                    if str(segID) not in seg_map[str(fname)]:
                        continue
                    sent = seg_map[str(fname)][str(segID)]
                    label = "" if label!=label else label
                    for lbl in label.split(","):
                        if lbl not in LABELS_OF_INTEREST:
                            lbl = ""
                        if lbl not in self.all_labels and lbl!="":
                            self.all_labels.append(lbl)
                        # lbl is a label of interest
                        # label is the whole list of labels
                        self.data_all.append((fname, segID, sent, lbl, label.split(","), lang_dir))
                        if lang_dir == lang: # and (lbl!="" or random.randint(0,5)<4): 
                            self.data_lang.append((fname, segID, sent, lbl, label.split(","), lang_dir))
            if lang_dir==lang and mode=="val" and cross_val:
                self.data_lang = self.data_lang[cross_val_split_idx::5]
            if lang_dir==lang and mode=="train" and cross_val:
                del self.data_lang[cross_val_split_idx::5]
 
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if MT_augment:
            if mode=="pretrain":
                # self.data_all = self.augment_data(self.data_all)
                self.data_all = self.augment_data_googletrans()
        if mode=="pretrain":
            print("Printing label count stats for pretrain")
            count = self.label_count_stats(self.data_all)
            print(count)
        elif mode=="train":
            print("Printing label count stats for train")
            count = self.label_count_stats(self.data_lang)
            print(count)

    def __len__(self):
        return len(self.data_all) if self.mode=="pretrain" else len(self.data_lang)
    def __getitem__(self, idx):
        fname, segID, txt1, propaganda, this_all_labels, ln = self.data_all[idx] if self.mode=="pretrain" else self.data_lang[idx]
        txt2 = propaganda if propaganda!="" else random.choice([x for x in self.all_labels if x not in this_all_labels])
        if random.choice([0,2])==0 and propaganda!="" and self.mode=="train":
            propaganda = ""
            txt2 = random.choice([x for x in LABELS_OF_INTEREST if x not in this_all_labels and x !=""])
        txt2_exp = txt2
        if use_def:
            txt2_exp = txt2+": "+LABELS_DEF[LABELS_DEF[0]==txt2][1].values[0]
        txt = self.tokenizer(txt1, txt2_exp, return_tensors="pt", truncation_strategy="only_first", pad_to_max_length=True, max_length=128)
        txt["input_ids"] = txt["input_ids"].squeeze(0).to(device)
        txt["attention_mask"] = txt["attention_mask"].squeeze(0).to(device)
        txt['token_type_ids'] = txt['token_type_ids'].squeeze(0).to(device)
        #TODO: swap back to 2 for bart-mnli!!!
        label = torch.tensor(0 if (propaganda=="" or propaganda not in this_all_labels) else 2).to(device)
        propaganda_idx = torch.tensor(self.all_labels.index(txt2))
        return fname, segID, txt, propaganda_idx, label