# SemEval23-Task-3-UIUC-Team


```
# train
python src/subtask3_CatByCat.py False
# Eval
python src/subtask3_CatByCat.py True

# run scorer:
python scorers/scorer-subtask-3.py -p baselines/our-dev-output-subtask3-it_def.txt -g data/it/dev-labels-subtask-3.txt --techniques_file_path scorers/techniques_subtask3.txt

```

# Results: 


Per-label performance: (c, c_fscore, c_precision, c_recall, sum(y_true_c))



## English

MarianMT
 - micro-F1=0.38898       macro-F1=0.29624

GoogleTrans pretrained 
- (3 eps) micro-F1=0.49635       macro-F1=0.40172 (ckpt)
- (6 eps) micro-F1=0.48050       macro-F1=0.40234 

With 3 eps of val data on pretrain+train checkpoint
 - micro-F1=0.44487       macro-F1=0.32080


## French

MarianMT
 - micro-F1=0.41096       macro-F1=0.13899

GoogleTrans pretrained 
 - (3 eps) micro-F1=0.40997       macro-F1=0.31876
 - (6 eps) micro-F1=0.42757       macro-F1=0.29934 (ckpt)

With 3 eps of val data on pretrain+train checkpoint
 - micro-F1=0.40907       macro-F1=0.24570

## German

MarianMT
 - micro-F1=0.40552       macro-F1=0.13567

GoogleTrans 
 - (3 eps) micro-F1=0.37641       macro-F1=0.25104
 - (6 eps) micro-F1=0.39173       macro-F1=0.26834 (ckpt)

With 3 eps of val data on pretrain+train checkpoint
 - micro-F1=0.43853       macro-F1=0.25394


## Italian:

MarianMT pretrained
 - micro-F1=0.49915       macro-F1=0.14364

GoogleTrans pretrained 
 - (3 eps) micro-F1=0.49203       macro-F1=0.23986
 - (6 eps) micro-F1=0.46704       macro-F1=0.24399 (ckpt)

With 3 eps of val data on pretrain+train checkpoint
 - micro-F1=0.51671       macro-F1=0.22125

## Polish

MarianMT
 - micro-F1=0.23499       macro-F1=0.08016

GoogleTrans
 - (3 eps) micro-F1=0.32061       macro-F1=0.21502
 - (6 eps) micro-F1=0.36923       macro-F1=0.24727 (ckpt)
 - (11 eps) micro-F1=0.37411       macro-F1=0.23668

With 3 eps of val data on pretrain+train checkpoint
 - micro-F1=0.28311       macro-F1=0.14006

po_to_en on English model: micro-F1=0.26642       macro-F1=0.18577

## Russian

MarianMT
 - micro-F1=0.40393       macro-F1=0.11573

GoogleTrans
 - (6 eps) micro-F1=0.41674       macro-F1=0.21749 (ckpt)

With 3 eps of val data on pretrain+train checkpoint
 - (3 eps) micro-F1=0.39090       macro-F1=0.15123


# Methods
- Cross-translated the dataset: [en, fr, it, ge, po, ru] all translated into each other. Using this to collect pretrain dataset
- Pretrained for 3 epochs and saved the checkpoint


## Side Note

German and italian test set is good
for the other four languages, submit using the previous checkpoints instead

Steps to translate surprise language to English and evaluate on English model checkpoint:
 - externally translate the test sets into English
 - create a temp copy of the original english test set
 - copy the translated content from es/gr/ka to this file `data/en/test-labels-subtask-3.template`
 - Do 0 epoch of pretrain/train/val/dev and just 1 epoch of test run
 - save result in the `baselines/submission` folder