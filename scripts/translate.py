import pandas as pd
from googletrans import Translator

# read in the CSV file
df = pd.read_csv("file.csv")

# select the column you want to translate
column_to_translate = df["column_name"]

# initialize the translator
translator = Translator()

# create a new list to store the translated text
translated_column = []

# translate each element in the column
for text in column_to_translate:
    translated = translator.translate(text, dest="fr").text #fr is the destination language
    translated_column.append(translated)

# add the translated column to the dataframe
df["translated_column_name"] = translated_column

# write the dataframe to a new CSV file
df.to_csv("translated_file.csv", index=False)
