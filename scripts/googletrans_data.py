import pandas as pd

path = "/Users/genglinliu/Documents/GitHub/SemEval23-Task-3-UIUC-Team/data_googletrans_augmented/data_text/en/train-labels-subtask-3.template"

df_en = pd.read_csv(path, sep='\t', names=["doc_id", "paragraph_id", "text"], skip_blank_lines=False, quoting=3)

print(df_en)