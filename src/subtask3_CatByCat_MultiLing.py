"""
TODO:
Add online sampling instead of in-advance
python scorers/scorer-subtask-3.py -p baselines/our-train-output-subtask3-en.txt -g data/en/train-labels-subtask-3.txt --techniques_file_path scorers/techniques_subtask3.txt
python scorers/scorer-subtask-3.py -p baselines/our-train-output-subtask3-en2.txt -g data/en/train-labels-subtask-32.txt --techniques_file_path scorers/techniques_subtask3.txt

python scorers/scorer-subtask-3.py -p baselines/our-train-output-subtask3-it2.txt -g data/it/train-labels-subtask-32.txt --techniques_file_path scorers/techniques_subtask3.txt
micro-F1=0.12850	macro-F1=0.11477


Just on Loaded_Language: micro-F1=0.69336	macro-F1=0.98667
On Loaded_Language,Repetition,Doubt: micro-F1=0.47684       macro-F1=0.92623
On "Loaded_Language","Name_Calling-Labeling","Repetition","Doubt": micro-F1=0.41938	macro-F1=0.89595
Above with definitions: micro-F1=0.47917	macro-F1=0.90273
On top 8: micro-F1=0.45370       macro-F1=0.79689
"""
import os
import sys
import copy
import json
import torch
import random
import pandas as pd
from datetime import datetime
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import MarianMTModel, MarianTokenizer


#accelerator = Accelerator()

## Initialize Settings
#lang = "en"
lang = "it"
lrate = 1e-5 #1e-6 has final loss of 0.1236
use_def = True 
MT_augment = True
skip_train = sys.argv[1].lower() == 'true'
cross_val = False 
device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = "data/"
#model_name = "xlm-mlm-xnli15-1024" 
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
if model_name == "t5-large":
    LABELS_OF_INTEREST = ['Appeal_to_Hypocrisy', 'Obfuscation-Vagueness-Confusion'] #, \
                          #'Whataboutism', 'Appeal_to_Popularity', 'Straw_Man']
LABELS_OF_INTEREST_pos_counter = {}
LABELS_OF_INTEREST_neg_counter = {}

LABELS_DEF = pd.read_csv("resources/task3_def.csv",header=None)

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
        self.tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp" if "T5" in model_name else model_name)
        if MT_augment:
            if mode=="pretrain":
                self.data_all = self.augment_data(self.data_all)
            if mode=="train":
                self.data_lang = self.augment_data(self.data_lang)

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
    def augment_data(self, data):
        print(data[1])
        new_data = data
        augment_data_cache, augmented_fname = [], "taskIII_augment_data_cache_"+self.mode+".json"
        if os.path.exists(augmented_fname):
            with open(augmented_fname, "r") as f:
                augment_data_cache = json.load(f)
            for x in augment_data_cache:
                if x not in data:
                    data.append(x)
            return data
        MT_ln_map = {"en":"en","fr":"fr","ge":"de","it":"it","po":"pl","ru":"ru"}
        for ln in ["en","fr","ge","it","ru"]:
            for src_ln in ["en","fr","ge","it","ru"]:
                if src_ln == ln:
                    continue
                ln, src_lang = MT_ln_map[ln], MT_ln_map[src_ln]
                self.mt_model_name = "Helsinki-NLP/opus-mt-"+src_lang+"-"+ln
                self.mt_model = MarianMTModel.from_pretrained(self.mt_model_name).to(device)
                self.mt_tokenizer = MarianTokenizer.from_pretrained(self.mt_model_name)
                self.bmt_model_name = "Helsinki-NLP/opus-mt-"+ln+"-"+src_lang
                self.bmt_model = MarianMTModel.from_pretrained(self.bmt_model_name).to(device)
                self.bmt_tokenizer = MarianTokenizer.from_pretrained(self.bmt_model_name)
                for data_x in copy.deepcopy(data):
                    if True and src_ln ==data_x[-1]:
                        if len(new_data)%2000==0:
                            with open(augmented_fname, "w") as f:
                                json.dump(augment_data_cache, f)
                            print(src_lang, ln, data_x[2], len(data), datetime.now())
                        txt_MT = self.translate(data_x[2], src_lang, ln)
                        data_x = data_x[:2]+(txt_MT,)+data_x[3:-1]+(ln,)
                        augment_data_cache.append(data_x)
                        txt_bMT = self.translate(txt_MT, ln, src_lang, True)
                        data_x = data_x[:2]+(txt_bMT,)+data_x[3:]
                        augment_data_cache.append(data_x)
                        new_data.append(data_x)
            if src_ln!=ln:
                break
        self.mt_model, self.bmt_model = None, None
        return new_data
    def translate(self, txt, src_lang, tgt_lang, backward=False):
        src_text = [">>"+tgt_lang+"<< "+txt]
        if backward:
            tokens = self.bmt_tokenizer(src_text, return_tensors="pt", pad_to_max_length=True, max_length=512)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            translated = self.bmt_model.generate(**tokens)
            return [self.bmt_tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]
        tokens = self.mt_tokenizer(src_text, return_tensors="pt", pad_to_max_length=True, max_length=512)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        translated = self.mt_model.generate(**tokens)
        return [self.mt_tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]

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