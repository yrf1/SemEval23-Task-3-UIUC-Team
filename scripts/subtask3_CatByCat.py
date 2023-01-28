"""
TODO:
Add online sampling instead of in-advance
python scorers/scorer-subtask-3.py -p baselines/our-train-output-subtask3-en.txt -g data/en/train-labels-subtask-3.txt --techniques_file_path scorers/techniques_subtask3.txt
python scorers/scorer-subtask-3.py -p baselines/our-train-output-subtask3-en2.txt -g data/en/train-labels-subtask-32.txt --techniques_file_path scorers/techniques_subtask3.txt

Just on Loaded_Language: micro-F1=0.69336	macro-F1=0.98667
On Loaded_Language,Repetition,Doubt: micro-F1=0.47684       macro-F1=0.92623
On "Loaded_Language","Name_Calling-Labeling","Repetition","Doubt": micro-F1=0.41938	macro-F1=0.89595
Above with definitions: micro-F1=0.47917	macro-F1=0.90273
On top 8: micro-F1=0.45370       macro-F1=0.79689
"""
import os
import torch
import random
import pandas as pd
from datetime import datetime
from collections import Counter
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5ForConditionalGeneration


accelerator = Accelerator()

## Initialize Settings
lang = "en"
lrate = 1e-6
use_def = True #
skip_train = True #
cross_val = False #
device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = "data/"
model_name = "facebook/bart-large-mnli"
#model_name, skip_train, cross_val, use_def, device = "t5-large", True, False, False, "cuda"
task_dir_train = "train-articles-subtask-3"
task_label_fname_train = "train-labels-subtask-3.txt"
task_dir_dev = "dev-articles-subtask-3"
task_label_fname_dev = "dev-labels-subtask-3.txt"
df = pd.read_csv("data/en/"+task_label_fname_train, sep="\t",header=None)[2].values
df = ["" if x!=x else x for x in df]
label_count = Counter([y for x in df for y in x.split(",")])
print(label_count)
LABELS_OF_INTEREST = ["","Loaded_Language","Name_Calling-Labeling","Repetition","Doubt", \
        'Exaggeration-Minimisation','Appeal_to_Fear-Prejudice', 'Flag_Waving', 'Causal_Oversimplification']
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
        task_dir = task_dir_train if mode in ["train","pretrain","val"] else task_dir_dev
        task_label_fname = task_label_fname_train if mode in \
                        ["train","pretrain","val"] else task_label_fname_dev
        self.all_labels = [] if mode not in ["val", "dev"] else all_labels
        cccount = 0
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
                        #if cccount < 20 or mode=="test":
                        #    if len(lbls_GT_here)>0 and mode=="val":
                        #        if lbl in lbls_GT_here:
                        #            print(d[0],d[1],lbls_GT_here)
                        #            cccount += 1
                        self.data_lang.append((d[0],d[1],d[2],lbl,lbls_GT_here))
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
                        self.data_all.append((fname, segID, sent, lbl, label.split(",")))
                        if lang_dir == lang: # and (lbl!="" or random.randint(0,5)<4): #3
                            self.data_lang.append((fname, segID, sent, lbl, label.split(",")))
            if lang_dir==lang and mode=="val" and cross_val:
                self.data_lang = self.data_lang[cross_val_split_idx::5]
            if lang_dir==lang and mode=="train" and cross_val:
                del self.data_lang[cross_val_split_idx::5]
        if mode=="val":
            self.data_all = self.data_all[cross_val_split_idx::5]
        if mode=="train":
            del self.data_all[cross_val_split_idx::5]
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp" if "T5" in model_name else model_name)
        #self.tokenizer = self.tokenizer.add_tokens(['<S>','<Q>','<A>'])
        #print(len([x for x in self.data_lang if x[-1]==""]))
        #print(len([x for x in self.data_lang if x[-1]!=""]))
        #print(len(self.data_lang))
    def __len__(self):
        return len(self.data_all) if self.mode=="pretrain" else len(self.data_lang)
    def __getitem__(self, idx):
        fname, segID, txt1, propaganda, this_all_labels = self.data_all[idx] if self.mode=="pretrain" else self.data_lang[idx]
        txt2 = propaganda if propaganda!="" else random.choice([x for x in self.all_labels if x not in this_all_labels])
        if model_name == "facebook/bart-large-mnli":
            txt2_exp = txt2
            if use_def:
                txt2_exp = txt2+": "+LABELS_DEF[LABELS_DEF[0]==txt2][1].values[0]
            txt = self.tokenizer(txt1, txt2_exp, return_tensors="pt", pad_to_max_length=True, max_length=128)
        elif model_name == "t5-large":
            with open("resources/task3_few_shot_ex/"+propaganda+".txt", "r") as f:
                txt = f.read()
            txt = txt.replace("<QUERY>", txt1)
            #if propaganda in this_all_labels:
            #    print("~~~~~~~~~~~~")
            #    print(txt)
            txt = self.tokenizer(txt, return_tensors="pt", pad_to_max_length=True, max_length=128)
        txt["input_ids"] = txt["input_ids"].squeeze(0).to(device)
        txt["attention_mask"] = txt["attention_mask"].squeeze(0).to(device)
        label = torch.tensor(0 if (propaganda=="" or propaganda not in this_all_labels) else 2).to(device)
        propaganda_idx = torch.tensor(self.all_labels.index(txt2))
        #print(fname, segID, txt1, label.item())
        return fname, segID, txt, propaganda_idx, label

#pretrain_dataset = MyDataset("pretrain") 
#pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=4, shuffle=True)

train_results_tracker, dev_results_tracker  = {}, {}
for cross_val_split_idx in range(5):
    print(cross_val_split_idx, datetime.now())
    train_dataset = MyDataset("train", cross_val_split_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataset = MyDataset("val", cross_val_split_idx, all_labels=train_dataset.all_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    #for idx, data in enumerate(val_dataset):
    #    pass
    dev_dataset = MyDataset("dev", all_labels=train_dataset.all_labels)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1)
    print(len(val_dataset), len(dev_dataset)) #18996 6254
    ## Set Model
    if model_name == "facebook/bart-large-mnli":
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    if model_name == "t5-large":
        model = T5ForConditionalGeneration.from_pretrained("bigscience/T0pp") #t5-large")
    optim = torch.optim.AdamW(model.parameters(), lr=lrate)
    loss = torch.nn.CrossEntropyLoss()
    ep = 4
    model_ckpts = "ckpts/"+str(cross_val_split_idx)+"/ep_"+str(ep)+"_NLI"+("_def" if use_def else "")+".pt"
    ## Train & Eval, ("pretrain",5,pretrain_dataloader)
    for (mode, tot_eps, dataloader) in [\
            ("train",8,train_dataloader), ("val",1 if (cross_val or \
            model_name == "t5-large") else 0,val_dataloader), ("dev",1,dev_dataloader)]:
        if skip_train and mode=="train": 
            if model_name == "facebook/bart-large-mnli":
                model.load_state_dict(torch.load(model_ckpts))
            continue
        model, optim, dataloader = accelerator.prepare(model, optim, dataloader)
        if mode in ["dev","val","test"]:
            model = model.eval()
        for ep in range(tot_eps):
            loss_tracker = []
            for idx, (fname, segID, x, prop_idx, y) in enumerate(dataloader):
                optim.zero_grad()
                if model_name == "facebook/bart-large-mnli":
                    out = model(**x)
                if mode in ["pretrain", "train"]:
                    loss_ = loss(out.logits, y)
                    loss_.backward()
                    loss_tracker.append(loss_.item())
                    optim.step()
                if mode in ["val","dev"]:
                    if model_name == "facebook/bart-large-mnli":
                        pred_y = out.logits.argmax(1).item()
                    elif model_name == "t5-large":
                        pred_y = model.generate(x["input_ids"], max_length=4)
                        pred_y = dataloader.dataset.tokenizer.decode(pred_y[0], skip_special_tokens=True)
                        print(pred_y, y)
                        pred_y = 2 if "yes" in pred_y.lower() else 0
                if mode in ["val"]:
                    if fname not in train_results_tracker:
                        train_results_tracker[fname] = {}
                    if segID not in train_results_tracker[fname]:
                        train_results_tracker[fname][segID] = []
                    if pred_y>1:
                        pred_y = dataloader.dataset.all_labels[prop_idx] 
                        train_results_tracker[fname][segID].append(pred_y)
                if mode in ["dev"]:
                    if fname not in dev_results_tracker:
                        dev_results_tracker[fname] = {}
                    if segID not in dev_results_tracker[fname]:
                        dev_results_tracker[fname][segID] =[]
                    if pred_y>1:
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
        data.append((fname[0], segID[0], ",".join(pred_y)))
dev_results_tracker = pd.DataFrame(data)
dev_results_tracker.to_csv("baselines/our-dev-output-subtask3-"+lang+("_def" if use_def else "")+".txt", \
    sep="\t", index=None, header=None)