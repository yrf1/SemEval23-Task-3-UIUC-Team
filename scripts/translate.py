import pandas as pd
import os
from tqdm import tqdm
from googletrans import Translator # pip install googletrans==3.1.0a0

# data_dir = "/shared/nas/data/users/genglin2/SemEval/SemEval23-Task-3-UIUC-Team/data" # server
data_dir = "/Users/genglinliu/Documents/GitHub/SemEval23-Task-3-UIUC-Team/data" # local

# initialize the translator
translator = Translator()

mt_map = {"en":"en","fr":"fr","ge":"de","it":"it","po":"pl","ru":"ru"}

# col_names=['doc_id', 'paragraph_id', 'text', 'extra_1', 'extra_2'] 
col_names=['doc_id', 'paragraph_id', 'text']
path = data_dir + "/po/train-labels-subtask-3.template"
df = pd.read_csv(path, sep='\t', names=col_names, on_bad_lines="warn")

for lang_dir in os.listdir(data_dir):
    col_names=['doc_id', 'paragraph_id', 'text', 'extra_1', 'extra_2'] 

    for mode in ["train", "dev"]:
        # ex. "en/train-labels-subtask-3.template"
        current_folder = lang_dir+"/"+mode+"-labels-subtask-3.template"
        df = pd.read_csv(data_dir+"/"+current_folder, sep="\t", names=col_names, header=None)

        # ex. you're at English folder right now, so translate English to the 5 other languages
        for target_lang in ["en", "fr", "de", "it", "pl", "ru"]:

            print("current folder: " + current_folder)
            print("Translating from: " + lang_dir + " to: " + target_lang)

            if target_lang == mt_map[lang_dir]:
                continue
            
            column_to_translate = df["text"]

            # create a new list to store the translated text
            translated_column = []

            # translate each element in the column
            for text in tqdm(column_to_translate):
                translated = translator.translate(text, dest=target_lang, src=mt_map[lang_dir]).text #fr is the destination language
                translated_column.append(translated)

            # add the translated column to the dataframe
            df[str("translated_"+target_lang)] = translated_column

        # write the dataframe to a new CSV file
        df.to_csv(data_dir+"/"+lang_dir+"/"+mode+"-translated.csv", sep="\t", index=False)