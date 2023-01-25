"""
TODO:
Add online sampling instead of in-advance
python scorers/scorer-subtask-3.py -p baselines/our-train-output-subtask3-en.txt -g data/en/train-labels-subtask-3.txt --techniques_file_path scorers/techniques_subtask3.txt
python scorers/scorer-subtask-3.py -p baselines/our-train-output-subtask3-en2.txt -g data/en/train-labels-subtask-32.txt --techniques_file_path scorers/techniques_subtask3.txt

python scorers/scorer-subtask-3.py -p baselines/our-train-output-subtask3-it2.txt -g data/it/train-labels-subtask-32.txt --techniques_file_path scorers/techniques_subtask3.txt
"""
import os
import torch
import random
import pandas as pd
from datetime import datetime
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

## Initialize Settings
#lang = "en"
lang = "it"
lrate = 1e-5 
use_def = True 
skip_train = False
cross_val = False 
device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = "data/"
model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7" # Train: micro-F1=0.03161	macro-F1=0.06228 (?) #"facebook/bart-large-mnli"
task_dir_train = "train-articles-subtask-3"
task_label_fname_train = "train-labels-subtask-3.txt"
task_dir_dev = "dev-articles-subtask-3"
task_label_fname_dev = "dev-labels-subtask-3.txt"
df = pd.read_csv("data/"+lang+"/"+task_label_fname_train, sep="\t",header=None)[2].values
df = ["" if x!=x else x for x in df]
label_count = Counter([y for x in df for y in x.split(",")])
print(label_count)
LABELS_OF_INTEREST = [k for k,v in label_count.items() if v>0] #100]
#LABELS_OF_INTEREST = ["","Loaded_Language","Name_Calling-Labeling","Repetition","Doubt", \
#        'Exaggeration-Minimisation','Appeal_to_Fear-Prejudice', 'Flag_Waving', 'Causal_Oversimplification']

LABELS_DEF = pd.read_csv("utils/task3_def.csv",header=None)

## Load Data
class MyDataset(Dataset):
    def __init__(self, mode="train", cross_val_split_idx=-1, all_labels=None):
        self.data_all, self.data_lang = [], []
        task_label_fname = task_label_fname_train if mode in \
                        ["train","pretrain","val"] else task_label_fname_dev
        self.all_labels = [] if mode not in ["val", "dev"] else all_labels
        for lang_dir in os.listdir(data_dir):
            labels = pd.read_csv(data_dir+"/"+lang_dir+"/"+task_label_fname, sep="\t", header=None) \
                     if mode in ["train","pretrain","val"] else None
            with open(data_dir+"/"+lang_dir+"/"+task_label_fname.replace(".txt",".template"), "r") as f:
                seg_data = f.read()
                seg_data = [x.split("\t") for x in seg_data.split("\n")[:-1]]
            seg_map = {}
            for d in seg_data:
                if d[0] not in seg_map:
                    seg_map[d[0]] = {}
                seg_map[d[0]][d[1]] = d[-1]
                if mode in ["val","dev"] and lang_dir==lang:
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
                        self.data_all.append((fname, segID, sent, lbl, label.split(","), lang_dir))
                        if lang_dir == lang: # and (lbl!="" or random.randint(0,5)<4): 
                            self.data_lang.append((fname, segID, sent, lbl, label.split(","), lang_dir))
            if lang_dir==lang and mode=="val" and cross_val:
                self.data_lang = self.data_lang[cross_val_split_idx::5]
            if lang_dir==lang and mode=="train" and cross_val:
                del self.data_lang[cross_val_split_idx::5]
        if mode=="val":
            pass
            #self.data_all = self.data_all[cross_val_split_idx::5]
        if mode=="train":
            del self.data_lang[cross_val_split_idx::5]
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.data_all) if self.mode=="pretrain" else len(self.data_lang)
    def __getitem__(self, idx):
        fname, segID, txt1, propaganda, this_all_labels, ln = self.data_all[idx] if self.mode=="pretrain" else self.data_lang[idx]
        txt2 = propaganda if propaganda!="" else random.choice([x for x in self.all_labels if x not in this_all_labels])
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
        #print(fname, segID, txt1, label.item())
        return fname, segID, txt, propaganda_idx, label


pretrain_dataset = MyDataset("pretrain") 
pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=8, shuffle=True)

train_results_tracker, dev_results_tracker  = {}, {}
for cross_val_split_idx in range(5):
    print(cross_val_split_idx, datetime.now())
    train_dataset = MyDataset("train", cross_val_split_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataset = MyDataset("val", cross_val_split_idx, all_labels=train_dataset.all_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    dev_dataset = MyDataset("dev", all_labels=train_dataset.all_labels)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1)
    print(len(train_dataset), len(val_dataset), len(dev_dataset)) #18996 6254
    ## Set Model
    #if model_name in ["facebook/bart-large-mnli","MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"]:
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    optim = torch.optim.AdamW(model.parameters(), lr=lrate)
    loss = torch.nn.CrossEntropyLoss()
    ep = 4
    model_ckpts = "ckpts/"+str(cross_val_split_idx)+"/ep_"+str(ep)+"_NLI_"+lang+("_def" if use_def else "")+".pt"
    if not os.path.exists("ckpts/"+str(cross_val_split_idx)):
        os.system("mkdir ckpts/"+str(cross_val_split_idx))
    if skip_train:
        print("loaded ckpt from... " + model_ckpts)
        model.load_state_dict(torch.load(model_ckpts))
    ## Train & Eval, ("pretrain",5,pretrain_dataloader)
    for (mode, tot_eps, dataloader) in [("pretrain",2 if not skip_train else 0,pretrain_dataloader),\
            ("train",2 if not skip_train else 0,train_dataloader), ("val",1,val_dataloader), ("dev",1,dev_dataloader)]:
        if skip_train and mode=="train": 
            if model_name == "facebook/bart-large-mnli":
                print("loaded ckpt from... " + model_ckpts)
                model.load_state_dict(torch.load(model_ckpts))
            continue
        #model, optim, dataloader = accelerator.prepare(model, optim, dataloader)
        if mode in ["dev","val","test"]:
            model = model.eval()
        for ep in range(tot_eps):
            loss_tracker = []
            for idx, (fname, segID, x, prop_idx, y) in enumerate(dataloader):
                optim.zero_grad()
                #if model_name in ["facebook/bart-large-mnli","MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"]:
                out = model(**x)
                if mode in ["pretrain", "train"]:
                    loss_ = loss(out.logits, y)
                    loss_.backward()
                    loss_tracker.append(loss_.item())
                    optim.step()
                if mode in ["val","dev"]:
                    #if model_name in ["facebook/bart-large-mnli","MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"]:
                    pred_y = out.logits.argmax(1).item()
                    
                if mode in ["val"]:
                    if fname not in train_results_tracker:
                        train_results_tracker[fname] = {}
                    if segID not in train_results_tracker[fname]:
                        train_results_tracker[fname][segID] = []
                    if pred_y>1: #TODO: 1 for bart-mNLI!!!
                        pred_y = dataloader.dataset.all_labels[prop_idx] 
                        train_results_tracker[fname][segID].append(pred_y)
                if mode in ["dev"]:
                    if fname not in dev_results_tracker:
                        dev_results_tracker[fname] = {}
                    if segID not in dev_results_tracker[fname]:
                        dev_results_tracker[fname][segID] =[]
                    if pred_y>1: #TODO: 1 for bart-mNLI!!!
                        pred_y = dataloader.dataset.all_labels[prop_idx] 
                  
                        dev_results_tracker[fname][segID].append(pred_y)
            if mode in ["pretrain", "train"]:
                print(sum(loss_tracker)/len(loss_tracker))
            if mode == "train":
                torch.save(model.state_dict(), model_ckpts)
    if not cross_val:
        break

data = []
for fname, v in train_results_tracker.items():
    for segID, pred_y in v.items():
        data.append((fname[0], segID[0], ",".join(pred_y)))
train_results_tracker = pd.DataFrame(data)
train_results_tracker.to_csv("baselines/our-train-output-subtask3-"+lang+("_def" if use_def else "")+".txt", \
    sep="\t", index=None, header=None)
data = []
for fname, v in dev_results_tracker.items():
    for segID, pred_y in v.items():
        pred_y = list(set(pred_y))
        data.append((fname[0], segID[0], ",".join(pred_y)))
dev_results_tracker = pd.DataFrame(data)
dev_results_tracker.to_csv("baselines/our-dev-output-subtask3-"+lang+("_def" if use_def else "")+".txt", \
    sep="\t", index=None, header=None)
