"""
TODO:
Add online sampling instead of in-advance
python scorers/scorer-subtask-1.py -p baselines/our-train-output-subtask1-en.txt -g data/en/train-labels-subtask-1.txt
"""
import os
import torch
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer


## Initialize Settings
lang = "en"
lrate = 5e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = "data/"
model_name = "xlm-roberta-base"
task_dir_train = "train-articles-subtask-1"
task_label_fname_train = "train-labels-subtask-1.txt"
task_dir_dev = "dev-articles-subtask-1"
task_label_fname_dev = "dev-labels-subtask-1.txt"
LABEL_MAP = {"opinion":0, "reporting":1, "satire":2, -1:-1}
LABEL_MAP_rv = {0:"opinion", 1:"reporting", 2:"satire"}

## Load Data
class MyDataset(Dataset):
    def __init__(self, mode="train", cross_val_split_idx=-1):
        self.data_all, self.data_lang = [], []
        task_dir = task_dir_train if mode in ["train","pretrain","val"] else task_dir_dev
        task_label_fname = task_label_fname_train if mode in \
                        ["train","pretrain","val"] else task_label_fname_dev
        for lang_dir in os.listdir(data_dir):
            labels = pd.read_csv(data_dir+"/"+lang_dir+"/"+task_label_fname, sep="\t", header=None) \
                     if mode in ["train","pretrain","val"] else None
            for fname in os.listdir(data_dir+"/"+lang_dir+"/"+task_dir):
                with open(data_dir+"/"+lang_dir+"/"+task_dir+"/"+fname, "r") as f:
                    txt = f.read()
                fname = fname.split(".txt")[0].split("article")[-1]
                label = labels[labels[0]==int(fname)][1].values[0] \
                        if mode in ["train","pretrain","val"] else -1
                self.data_all.append((fname, txt,label))
                if lang_dir == lang:
                    self.data_lang.append((fname, txt,label))
        if mode in ["train","pretrain","val"]:
            data_all_opinion = [x for x in self.data_all if x[-1]=="opinion"]
            data_all_reporting = [x for x in self.data_all if x[-1]=="reporting"]
            data_all_satire = [x for x in self.data_all if x[-1]=="satire"]
            if mode=="val":
                data_all_opinion = data_all_opinion[cross_val_split_idx::5]
                data_all_reporting = data_all_reporting[cross_val_split_idx::5]
                data_all_satire = data_all_satire[cross_val_split_idx::5]
                self.data_all = data_all_opinion + data_all_reporting + data_all_satire
            if mode=="train":
                del data_all_opinion[cross_val_split_idx::5]
                del data_all_reporting[cross_val_split_idx::5]
                del data_all_satire[cross_val_split_idx::5]
                size_min_class = int(2.25*len(data_all_satire))
                self.data_all = random.sample(data_all_opinion,min(size_min_class,len(data_all_opinion))) + \
                            random.sample(data_all_reporting,min(size_min_class,len(data_all_reporting))) + \
                            random.sample(data_all_satire,min(size_min_class,len(data_all_satire)))
            data_lang_opinion = [x for x in self.data_lang if x[-1]=="opinion"]
            data_lang_reporting = [x for x in self.data_lang if x[-1]=="reporting"]
            data_lang_satire = [x for x in self.data_lang if x[-1]=="satire"]
            if mode=="val":
                data_lang_opinion = data_lang_opinion[cross_val_split_idx::5]
                data_lang_reporting = data_lang_reporting[cross_val_split_idx::5]
                data_lang_satire = data_lang_satire[cross_val_split_idx::5]
                self.data_lang = data_lang_opinion + data_lang_reporting + data_lang_satire
            if mode=="train":
                del data_lang_opinion[cross_val_split_idx::5]
                del data_lang_reporting[cross_val_split_idx::5]
                del data_lang_satire[cross_val_split_idx::5]
                size_min_class = int(2.25*len(data_lang_satire))
                self.data_lang = random.sample(data_lang_opinion,min(size_min_class,len(data_lang_opinion))) + \
                            random.sample(data_lang_reporting,min(size_min_class,len(data_lang_reporting))) + \
                            random.sample(data_lang_satire,min(size_min_class,len(data_lang_satire)))
        self.fnames_lang = [x[0] for x in self.data_lang]
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    def __len__(self):
        return len(self.data_all) if self.mode=="pretrain" else len(self.data_lang)
    def __getitem__(self, idx):
        _, txt, label = self.data_all[idx] if self.mode=="pretrain" else self.data_lang[idx]
        print(txt)
        quit()
        txt = self.tokenizer(txt, return_tensors="pt", pad_to_max_length=True, max_length=512)
        txt["input_ids"] = txt["input_ids"].squeeze(0).to(device)
        txt["attention_mask"] = txt["attention_mask"].squeeze(0).to(device)
        label = torch.tensor(LABEL_MAP[label]).to(device)
        return txt, label, idx

pretrain_dataset = MyDataset("pretrain") 
pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=4, shuffle=True)
dev_dataset = MyDataset("dev")
dev_dataloader = DataLoader(dev_dataset, batch_size=1)

train_results_tracker, dev_results_tracker  = [], []
for cross_val_split_idx in range(5):
    train_dataset = MyDataset("train", cross_val_split_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataset = MyDataset("val", cross_val_split_idx)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    ## Set Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=3).to(device)
    model.roberta.encoder.layer[11].output.dropout.p=0.5
    optim = AdamW(model.parameters(), lr=lrate)
    loss = torch.nn.CrossEntropyLoss()
    ## Train & Eval
    for (mode, tot_eps, dataloader) in [("pretrain",5,pretrain_dataloader), \
            ("train",5,train_dataloader), ("val",1,val_dataloader), ("dev",1,dev_dataloader)]:
        if mode in ["dev","val","test"]:
            model = model.eval()
        for ep in range(tot_eps):
            loss_tracker = []
            for idx, (x,y,fname_idx) in enumerate(dataloader):
                optim.zero_grad()
                out = model(**x)
                if mode in ["pretrain", "train"]:
                    loss_ = loss(out.logits, y)
                    loss_.backward()
                    loss_tracker.append(loss_.item())
                    optim.step()
                if mode in ["val"]:
                    pred_y = LABEL_MAP_rv[out.logits.argmax(1).item()]
                    train_results_tracker.append((dataloader.dataset.fnames_lang[fname_idx],pred_y))
                if mode in ["dev"]:
                    pred_y = LABEL_MAP_rv[out.logits.argmax(1).item()]
                    dev_results_tracker.append((dataloader.dataset.fnames_lang[fname_idx],pred_y))
        #    #if mode in ["pretrain", "train"]:
        #    #    print(ep, sum(loss_tracker)/len(loss_tracker))
train_results_tracker = pd.DataFrame(train_results_tracker)
train_results_tracker.to_csv("baselines/our-train-output-subtask1-"+lang+".txt", \
    sep="\t", index=None, header=None)
dev_results_tracker = pd.DataFrame(dev_results_tracker)
dev_results_tracker.to_csv("baselines/our-dev-output-subtask1-"+lang+".txt", \
    sep="\t", index=None, header=None)



