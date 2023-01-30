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

## Italian:

MarianMT pretrained
 - micro-F1=0.49915       macro-F1=0.14364

GoogleTrans pretrained 
 - (3 eps) micro-F1=0.49203       macro-F1=0.23986
 - (6 eps) micro-F1=0.46704       macro-F1=0.24399


## English

MarianMT
 - micro-F1=0.38898       macro-F1=0.29624

GoogleTrans pretrained 
- (3 eps) micro-F1=0.49635       macro-F1=0.40172
- (6 eps) micro-F1=0.48050       macro-F1=0.40234


## French

MarianMT
 - micro-F1=0.41096       macro-F1=0.13899

GoogleTrans pretrained 
 - (3 eps) micro-F1=0.40997       macro-F1=0.31876
 - (6 eps) micro-F1=0.42757       macro-F1=0.29934


## German

MarianMT
 - micro-F1=0.40552       macro-F1=0.13567

GoogleTrans 
 - (3 eps) micro-F1=0.37641       macro-F1=0.25104
 - (6 eps) micro-F1=0.39173       macro-F1=0.26834


## Polish

MarianMT
 - micro-F1=0.23499       macro-F1=0.08016

GoogleTrans
 - (3 eps) micro-F1=0.32061       macro-F1=0.21502
 - (6 eps) micro-F1=0.36923       macro-F1=0.24727

## Russian

MarianMT
 - micro-F1=0.40393       macro-F1=0.11573

GoogleTrans
 - (6 eps) micro-F1=0.41674       macro-F1=0.21749
