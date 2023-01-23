import pandas as pd
import os
from tqdm import tqdm
from googletrans import Translator # pip install googletrans==3.1.0a0
from google_trans_new import google_translator


# data_dir = "data/"
# for lang_dir in os.listdir(data_dir):
#     labels = pd.read_csv(data_dir+"/"+lang_dir+"/"+"train-labels-subtask-3.template", sep="\t", header=None)

# read in the tab separated template file
col_names=['doc_id', 'paragraph_id', 'text'] 
data_path = "/shared/nas/data/users/genglin2/SemEval/SemEval23-Task-3-UIUC-Team/data/en/dev-labels-subtask-3.template"
df = pd.read_csv(data_path, sep='\t', names=col_names)

# select the column you want to translate
column_to_translate = df["text"]

# initialize the translator
translator = Translator()
# translator = google_translator()

# create a new list to store the translated text
translated_column = []

# translate each element in the column
for text in tqdm(column_to_translate):
    translated = translator.translate(text, dest='fr', src='en').text #fr is the destination language
    translated_column.append(translated)

# add the translated column to the dataframe
df["translated_column_name"] = translated_column

print(df)

# write the dataframe to a new CSV file
df.to_csv("translated_file.csv", index=False)
