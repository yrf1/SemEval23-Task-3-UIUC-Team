# SemEval23-Task-3-UIUC-Team


```
# train
python src/subtask3_CatByCat.py False
# Eval
python src/subtask3_CatByCat.py True
```

|       | SubTask II |  SubTask III |
| ----------- | ----------- | ----------- |
| Baseline      | 0.605       | 0.16125 |
| Bart-mnli-wDef   | 0.71       | 0.34955 |
| SOTA   | 0.785        | 0.3779|

## Data
 - `data_all` contains the final combined datasets for each language that we use to train models
 - `/data_augmented` contains the original PLUS the cross-translated datasets
 - `/data_extra` contains the 2021 data and the extra part of 2023 v4 data that we are currently not using
 - `/data_original` is the origianl v2 dataset that we downloaded, and is used to augment the translated files

