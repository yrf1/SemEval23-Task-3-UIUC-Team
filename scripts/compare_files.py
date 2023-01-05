import os, sys

# Open a file
path_2023 = "data/en/train-articles-subtask-3"
path_2020 = "data/SemEval2020/datasets/train-articles"

dirs_2020 = os.listdir(path_2020)
dirs_2023 = os.listdir(path_2023)

# print((set(dirs_2023) - set(dirs_2020)) == (set(dirs_2020) - set(dirs_2023)) )

print(len(dirs_2020)) # 371
print(len(dirs_2023)) # 446

print(len(set(dirs_2023) - set(dirs_2020)))