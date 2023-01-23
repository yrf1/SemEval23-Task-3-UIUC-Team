import pandas as pd
from tqdm import tqdm
from googletrans import Translator # pip install googletrans==3.1.0a0

# data_dir = "/shared/nas/data/users/genglin2/SemEval/SemEval23-Task-3-UIUC-Team/data" # server
data_dir = "/Users/genglinliu/Documents/GitHub/SemEval23-Task-3-UIUC-Team/data" # local

# initialize the translator
translator = Translator()

mt_map = {"en":"en","fr":"fr","ge":"de","it":"it","po":"pl","ru":"ru"}
col_names=['doc_id', 'paragraph_id', 'text']

target_lang = "en"

# ex. we translate to English from the 5 other languages
for src_lang in ["en", "fr", "ge", "it", "po", "ru"]:   

    if target_lang == src_lang:
        continue 

    for mode in ["train", "dev"]:
        # ex. "en/train-labels-subtask-3.template"
        current_folder = src_lang+"/"+mode+"-labels-subtask-3.template"
        df = pd.read_csv(data_dir+"/"+current_folder, sep="\t", names=col_names, on_bad_lines="skip", header=None)

        print("current folder: " + current_folder)
        print("Translating from: " + src_lang + " to: " + target_lang)
        
        column_to_translate = df["text"]

        # create a new list to store the translated text
        translated_column = []

        # translate each element in the column
        for text in tqdm(column_to_translate):
            translated = translator.translate(text, dest=mt_map[target_lang], src=mt_map[src_lang]).text #fr is the destination language
            translated_column.append(translated)

        # add the translated column to the dataframe
        new_col_name = str(src_lang+"-"+target_lang)
        df[new_col_name] = translated_column

        df_out = df[['doc_id', 'paragraph_id', new_col_name]]

        # write the dataframe to a new CSV file
        df_out.to_csv(src_lang+"-"+target_lang+"-"+mode+"-translated.csv", sep="\t", index=False)