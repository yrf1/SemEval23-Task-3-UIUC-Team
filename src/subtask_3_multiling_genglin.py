"""
Running scorer locally:
python scorers/scorer-subtask-3.py -p baselines/googletrans-dev-output-subtask3-it_def.txt -g data/it/dev-labels-subtask-3.txt --techniques_file_path scorers/techniques_subtask3.txt
"""
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

## Initialize Settings
lang = "po" # NOTE: check eps
lrate = 1e-5 
BATCH_SIZE = 16
use_def = True 
MT_augment = True
# skip_train = sys.argv[1].lower() == 'true'
skip_pretrain = True
cross_val = False
device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = "data/"
model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7" 
task_label_fname_train = "train-labels-subtask-3.txt"
task_label_fname_dev = "dev-labels-subtask-3.txt"
df = pd.read_csv("data/"+lang+"/"+task_label_fname_train, sep="\t",header=None)[2].values
df = ["" if x!=x else x for x in df]
label_count = Counter([y for x in df for y in x.split(",")])
print(label_count)
# LABELS_OF_INTEREST = [k for k,v in label_count.items() if v>=0] #100]
# LABELS_OF_INTEREST is all of the labels right now
LABELS_OF_INTEREST = ["", "Appeal_to_Authority", "Appeal_to_Popularity", "Appeal_to_Values", "Appeal_to_Fear-Prejudice", "Flag_Waving", "Causal_Oversimplification", \
    "False_Dilemma-No_Choice", "Consequential_Oversimplification", "Straw_Man", "Red_Herring", "Whataboutism", "Slogans", "Appeal_to_Time", \
    "Conversation_Killer", "Loaded_Language", "Repetition", "Exaggeration-Minimisation", "Obfuscation-Vagueness-Confusion", "Name_Calling-Labeling", \
    "Doubt", "Guilt_by_Association", "Appeal_to_Hypocrisy", "Questioning_the_Reputation"]

LABELS_DEF = pd.read_csv("resources/task3_def.csv",header=None)

## Load Data
class MyDataset(Dataset):
    def __init__(self, mode="train", cross_val_split_idx=-1, all_labels=None):
        self.data_all, self.data_lang = [], []
        task_label_fname = task_label_fname_train if mode in \
                        ["train","pretrain","val"] else task_label_fname_dev
        self.all_labels = [] if mode not in ["val", "dev"] else all_labels
        for lang_dir in ["en", "it","ge","fr","po","ru"]: #os.listdir(data_dir):
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

    def augment_data_googletrans(self):

        data_text = "/shared/nas/data/users/genglin2/SemEval/SemEval23-Task-3-UIUC-Team/data_googletrans_augmented/data_text/"
        label_path = "/shared/nas/data/users/genglin2/SemEval/SemEval23-Task-3-UIUC-Team/data_googletrans_augmented/labels/train_labels_all.txt"

        def get_augmented_df(lang):

            df_lang = pd.read_csv(data_text+lang+"/"+"train-labels-subtask-3.template", \
                                sep='\t', \
                                names=["doc_id", "paragraph_id", "text"], \
                                dtype = {'doc_id': str, 'paragraph_id': str, 'text': str}, \
                                on_bad_lines='skip', \
                                quoting=3)

            df_labels = pd.read_csv(label_path, \
                                    sep='\t', \
                                    names=["doc_id", "paragraph_id", "labels"], \
                                    dtype = {'doc_id': str, 'paragraph_id': str, 'labels': str},\
                                    on_bad_lines='skip', \
                                    quoting=3)

            # merge lang df with labels on two columns
            df_res_lang = pd.merge(df_lang, df_labels, on=['doc_id','paragraph_id'])

            # transform on the merged df
            df = df_res_lang
            df["labels"] = df['labels'].str.split(',')
            df = df.assign(lbl=df['labels']).explode("lbl")
            df = df.assign(language=lang)
            df = df.loc[:, ["doc_id","paragraph_id","text","lbl", "labels", "language"]]

            return df

        df_en = get_augmented_df("en")
        df_fr = get_augmented_df("fr")
        df_ge = get_augmented_df("ge")
        df_it = get_augmented_df("it")
        df_po = get_augmented_df("po")
        df_ru = get_augmented_df("ru")
        df_all = pd.concat([df_en, df_fr, df_ge, df_it, df_po, df_ru], ignore_index=True, axis=0)
        df_res = df_all.fillna("")

        return df_res.values.tolist()

    def augment_data(self, data):
        new_data = data
        label_count = {}
        for d in data:
            if d[-3] not in label_count and d[-3]!="":
                label_count[d[-3]] = 0
            if d[-3]!="":
                label_count[d[-3]] += 1
        max_label_count = max(label_count.values())
        augment_data_cache, augmented_fname = [], "taskIII_augment_data_cache_"+self.mode+".json"
        if os.path.exists(augmented_fname):
            with open(augmented_fname, "r") as f:
                augment_data_cache = json.load(f)
            for x in augment_data_cache:
                if x not in data:
                    if x[-3]!="" and (self.mode=="pretrain" or x[-1]==lang):
                        if x[-3] not in label_count:
                            continue
                        if label_count[x[-3]] < max_label_count:
                            data.append(x)
                            label_count[x[-3]] += 1
            print(label_count)
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
                        data_x = data_x[:2]+(txt_bMT,)+data_x[3:-1]+(src_ln,)
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
    def label_count_stats(self, data):
        return Counter([x[-3] for x in data])

pretrain_dataset = MyDataset("pretrain") 
pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True)
print("pretrain length: ", len(pretrain_dataset))

train_results_tracker, dev_results_tracker  = {}, {}
for cross_val_split_idx in range(5):
    print(cross_val_split_idx, datetime.now())
    train_dataset = MyDataset("train", cross_val_split_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = MyDataset("val", cross_val_split_idx, all_labels=train_dataset.all_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    dev_dataset = MyDataset("dev", all_labels=train_dataset.all_labels)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1)
    print(len(pretrain_dataset), len(train_dataset), len(val_dataset), len(dev_dataset))
    ## Set Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lrate)
    loss = torch.nn.CrossEntropyLoss()
    model_ckpts_pretrain = "ckpts/"+str(cross_val_split_idx)+"/PRETRAIN_def.pt"

    if not os.path.exists("ckpts/"+str(cross_val_split_idx)):
        os.system("mkdir ckpts/"+str(cross_val_split_idx))
    if skip_pretrain:
        print("loaded ckpt from... " + model_ckpts_pretrain)
        model.load_state_dict(torch.load(model_ckpts_pretrain))
    print("start train")
    for (mode, tot_eps, dataloader) in [("pretrain",3 if not skip_pretrain else 0,pretrain_dataloader),\
            ("train",6,train_dataloader), ("dev",1,dev_dataloader)]:
        if skip_pretrain and mode=="pretrain": 
            continue
        if mode in ["dev","val","test"]:
            model = model.eval()
        for ep in range(tot_eps):
            loss_tracker = []
            for idx, (fname, segID, x, prop_idx, y) in tqdm(enumerate(dataloader)):
                optim.zero_grad()
                out = model(**x)
                if mode in ["pretrain", "train"]:
                    loss_ = loss(out.logits, y)
                    loss_.backward()
                    loss_tracker.append(loss_.item())
                    optim.step()
                if mode in ["val","dev"]:
                    pred_y = out.logits.argmax(1).item()
                if mode in ["val"]:
                    if fname not in train_results_tracker:
                        train_results_tracker[fname] = {}
                    if segID not in train_results_tracker[fname]:
                        train_results_tracker[fname][segID] = []
                    if pred_y>1: # 1 for bart-mNLI!!!
                        pred_y = dataloader.dataset.all_labels[prop_idx] 
                        train_results_tracker[fname][segID].append(pred_y)
                if mode in ["dev"]:
                    if fname not in dev_results_tracker:
                        dev_results_tracker[fname] = {}
                    if segID not in dev_results_tracker[fname]:
                        dev_results_tracker[fname][segID] =[]
                    if pred_y>1: # 1 for bart-mNLI!!!
                        pred_y = dataloader.dataset.all_labels[prop_idx] 
                        dev_results_tracker[fname][segID].append(pred_y)
            if mode in ["pretrain", "train"]:
                print(datetime.now(), sum(loss_tracker)/len(loss_tracker))
            if mode == "pretrain":
                torch.save(model.state_dict(), model_ckpts_pretrain)
            if mode == "train":
                model_ckpts_train = "ckpts/"+str(cross_val_split_idx)+"/ep_"+str(ep)+"_NLI_"+lang+("_def_googletrans" if use_def else "")+".pt"
                # model_ckpts_train = "ckpts/"+str(cross_val_split_idx)+"/ep_11"+"_NLI_"+lang+("_def_googletrans" if use_def else "")+".pt"
                torch.save(model.state_dict(), model_ckpts_train)
    if not cross_val:
        break

data = []
for fname, v in dev_results_tracker.items():
    for segID, pred_y in v.items():
        pred_y = list(set(pred_y))
        data.append((fname[0], segID[0], ",".join(pred_y)))
dev_results_tracker = pd.DataFrame(data)
dev_results_tracker.to_csv("baselines/googletrans-dev-output-subtask3-"+lang+("_def" if use_def else "")+".txt", \
    sep="\t", index=None, header=None)
