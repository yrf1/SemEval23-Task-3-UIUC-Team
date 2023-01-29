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

### GoogleTrans pretrained

micro-F1=0.49203       macro-F1=0.23986
```
('Whataboutism', 0.0, 0.0, 0.0, 1)
('Red_Herring', 0.0, 0.0, 0.0, 4)
('Obfuscation-Vagueness-Confusion', 0.0, 0.0, 0.0, 4)
('Consequential_Oversimplification', 0.0, 0.0, 0.0, 9)
('Flag_Waving', 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 12)
('Causal_Oversimplification', 0.0, 0.0, 0.0, 12)
('Straw_Man', 0.0, 0.0, 0.0, 15)
('Repetition', 0.1111111111111111, 0.3333333333333333, 0.06666666666666667, 15)
('False_Dilemma-No_Choice', 0.19999999999999998, 0.21428571428571427, 0.1875, 16)
('Appeal_to_Time', 0.32, 0.4444444444444444, 0.25, 16)
('Appeal_to_Popularity', 0.09523809523809525, 0.3333333333333333, 0.05555555555555555, 18)
('Slogans', 0.30188679245283023, 0.24242424242424243, 0.4, 20)
('Guilt_by_Association', 0.3428571428571428, 0.46153846153846156, 0.2727272727272727, 22)
('Appeal_to_Authority', 0.1643835616438356, 0.12244897959183673, 0.25, 24)
('Appeal_to_Hypocrisy', 0.058823529411764705, 0.14285714285714285, 0.037037037037037035, 27)
('Exaggeration-Minimisation', 0.21951219512195125, 0.15517241379310345, 0.375, 48)
('Appeal_to_Values', 0.4228571428571429, 0.30833333333333335, 0.6727272727272727, 55)
('Conversation_Killer', 0.29775280898876405, 0.18466898954703834, 0.7681159420289855, 69)
('Appeal_to_Fear-Prejudice', 0.4220779220779221, 0.2927927927927928, 0.7558139534883721, 86)
('Questioning_the_Reputation', 0.39094650205761317, 0.260989010989011, 0.7786885245901639, 122)
('Name_Calling-Labeling', 0.5260029717682021, 0.3597560975609756, 0.9779005524861878, 181)
('Doubt', 0.6394557823129251, 0.47394957983193275, 0.9825783972125436, 287)
('Loaded_Language', 0.670547147846333, 0.5115452930728241, 0.972972972972973, 296)
```

### MarianMT pretrained

micro-F1=0.49915       macro-F1=0.14364
```
('Whataboutism', 0.0, 0.0, 0.0, 1)
('Red_Herring', 0.0, 0.0, 0.0, 4)
('Obfuscation-Vagueness-Confusion', 0.0, 0.0, 0.0, 4)
('Consequential_Oversimplification', 0.0, 0.0, 0.0, 9)
('Flag_Waving', 0.0, 0.0, 0.0, 12)
('Causal_Oversimplification', 0.0, 0.0, 0.0, 12)
('Straw_Man', 0.0, 0.0, 0.0, 15)
('Repetition', 0.0, 0.0, 0.0, 15)
('False_Dilemma-No_Choice', 0.0, 0.0, 0.0, 16)
('Appeal_to_Time', 0.0, 0.0, 0.0, 16)
('Appeal_to_Popularity', 0.0, 0.0, 0.0, 18)
('Slogans', 0.0, 0.0, 0.0, 20)
('Guilt_by_Association', 0.2666666666666666, 0.2608695652173913, 0.2727272727272727, 22)
('Appeal_to_Authority', 0.0, 0.0, 0.0, 24)
('Appeal_to_Hypocrisy', 0.0, 0.0, 0.0, 27)
('Exaggeration-Minimisation', 0.15, 0.1875, 0.125, 48)
('Appeal_to_Values', 0.16438356164383564, 0.3333333333333333, 0.10909090909090909, 55)
('Conversation_Killer', 0.18181818181818185, 0.24390243902439024, 0.14492753623188406, 69)
('Appeal_to_Fear-Prejudice', 0.3712121212121212, 0.2752808988764045, 0.5697674418604651, 86)
('Questioning_the_Reputation', 0.26728110599078336, 0.30526315789473685, 0.23770491803278687, 122)
('Name_Calling-Labeling', 0.579957356076759, 0.4722222222222222, 0.7513812154696132, 181)
('Doubt', 0.6497175141242937, 0.5463182897862233, 0.8013937282229965, 287)
('Loaded_Language', 0.6727272727272726, 0.5464135021097046, 0.875, 296)
```

## English

micro-F1=0.38898       macro-F1=0.29624
```
('Appeal_to_Values', 0.0, 0.0, 0.0, 0)
('Consequential_Oversimplification', 0.0, 0.0, 0.0, 0)
('Appeal_to_Time', 0.0, 0.0, 0.0, 0)
('Questioning_the_Reputation', 0.0, 0.0, 0.0, 0)
('Whataboutism', 0.0, 0.0, 0.0, 2)
('Guilt_by_Association', 0.0, 0.0, 0.0, 4)
('Appeal_to_Hypocrisy', 0.0, 0.0, 0.0, 8)
('Straw_Man', 0.0, 0.0, 0.0, 9)
('Obfuscation-Vagueness-Confusion', 0.0, 0.0, 0.0, 13)
('Red_Herring', 0.0, 0.0, 0.0, 19)
('Causal_Oversimplification', 0.128, 0.07920792079207921, 0.3333333333333333, 24)
('Conversation_Killer', 0.0, 0.0, 0.0, 25)
('Appeal_to_Authority', 0.0, 0.0, 0.0, 28)
('Slogans', 0.0, 0.0, 0.0, 28)
('Appeal_to_Popularity', 0.0, 0.0, 0.0, 34)
('False_Dilemma-No_Choice', 0.0, 0.0, 0.0, 63)
('Flag_Waving', 0.47656249999999994, 0.38125, 0.6354166666666666, 96)
('Exaggeration-Minimisation', 0.2731707317073171, 0.18983050847457628, 0.48695652173913045, 115)
('Appeal_to_Fear-Prejudice', 0.40259740259740256, 0.36257309941520466, 0.45255474452554745, 137)
('Repetition', 0.16414686825053995, 0.11801242236024845, 0.2695035460992908, 141)
('Doubt', 0.3051470588235294, 0.23249299719887956, 0.44385026737967914, 187)
('Name_Calling-Labeling', 0.4861878453038673, 0.37130801687763715, 0.704, 250)
('Loaded_Language', 0.5777027027027026, 0.48787446504992865, 0.7080745341614907, 483)
```

## French

micro-F1=0.41096       macro-F1=0.13899
```
('Red_Herring', 0.0, 0.0, 0.0, 9)
('Flag_Waving', 0.0, 0.0, 0.0, 10)
('Whataboutism', 0.0, 0.0, 0.0, 12)
('Appeal_to_Time', 0.0, 0.0, 0.0, 14)
('Appeal_to_Popularity', 0.0, 0.0, 0.0, 17)
('Repetition', 0.0, 0.0, 0.0, 21)
('Straw_Man', 0.0, 0.0, 0.0, 23)
('Slogans', 0.0, 0.0, 0.0, 27)
('False_Dilemma-No_Choice', 0.0, 0.0, 0.0, 29)
('Guilt_by_Association', 0.17142857142857143, 0.5, 0.10344827586206896, 29)
('Obfuscation-Vagueness-Confusion', 0.0, 0.0, 0.0, 36)
('Appeal_to_Hypocrisy', 0.0, 0.0, 0.0, 37)
('Appeal_to_Authority', 0.0, 0.0, 0.0, 40)
('Appeal_to_Values', 0.13559322033898305, 0.26666666666666666, 0.09090909090909091, 44)
('Causal_Oversimplification', 0.0, 0.0, 0.0, 44)
('Conversation_Killer', 0.1643835616438356, 0.2857142857142857, 0.11538461538461539, 52)
('Consequential_Oversimplification', 0.0, 0.0, 0.0, 53)
('Appeal_to_Fear-Prejudice', 0.38235294117647056, 0.35135135135135137, 0.41935483870967744, 62)
('Exaggeration-Minimisation', 0.4042553191489361, 0.3333333333333333, 0.5135135135135135, 74)
('Questioning_the_Reputation', 0.375, 0.47368421052631576, 0.3103448275862069, 87)
('Doubt', 0.3393939393939394, 0.4, 0.29473684210526313, 95)
('Name_Calling-Labeling', 0.4808743169398907, 0.6567164179104478, 0.3793103448275862, 116)
('Loaded_Language', 0.7435064935064936, 0.6256830601092896, 0.916, 250)
```

## German

micro-F1=0.40552       macro-F1=0.13567
```
('Straw_Man', 0.0, 0.0, 0.0, 2)
('Red_Herring', 0.0, 0.0, 0.0, 4)
('Repetition', 0.0, 0.0, 0.0, 4)
('False_Dilemma-No_Choice', 0.0, 0.0, 0.0, 5)
('Appeal_to_Time', 0.0, 0.0, 0.0, 6)
('Consequential_Oversimplification', 0.0, 0.0, 0.0, 12)
('Whataboutism', 0.0, 0.0, 0.0, 13)
('Appeal_to_Popularity', 0.0, 0.0, 0.0, 17)
('Flag_Waving', 0.0, 0.0, 0.0, 18)
('Causal_Oversimplification', 0.0, 0.0, 0.0, 20)
('Obfuscation-Vagueness-Confusion', 0.0, 0.0, 0.0, 22)
('Guilt_by_Association', 0.33333333333333337, 0.46153846153846156, 0.2608695652173913, 23)
('Conversation_Killer', 0.10714285714285714, 0.12, 0.0967741935483871, 31)
('Appeal_to_Authority', 0.0, 0.0, 0.0, 36)
('Appeal_to_Values', 0.4583333333333333, 0.9166666666666666, 0.3055555555555556, 36)
('Slogans', 0.0, 0.0, 0.0, 39)
('Exaggeration-Minimisation', 0.07272727272727272, 0.16666666666666666, 0.046511627906976744, 43)
('Appeal_to_Fear-Prejudice', 0.41509433962264153, 0.36065573770491804, 0.4888888888888889, 45)
('Appeal_to_Hypocrisy', 0.0, 0.0, 0.0, 56)
('Loaded_Language', 0.26206896551724135, 0.27941176470588236, 0.24675324675324675, 77)
('Questioning_the_Reputation', 0.37125748502994005, 0.3563218390804598, 0.3875, 80)
('Doubt', 0.39766081871345027, 0.4358974358974359, 0.3655913978494624, 93)
('Name_Calling-Labeling', 0.7027027027027026, 0.6190476190476191, 0.8125, 240)
```

## Polish
micro-F1=0.23499       macro-F1=0.08016

```
('Straw_Man', 0.0, 0.0, 0.0, 3)
('Whataboutism', 0.0, 0.0, 0.0, 3)
('Causal_Oversimplification', 0.0, 0.0, 0.0, 5)
('Appeal_to_Time', 0.0, 0.0, 0.0, 5)
('Red_Herring', 0.0, 0.0, 0.0, 7)
('Slogans', 0.0, 0.0, 0.0, 7)
('False_Dilemma-No_Choice', 0.0, 0.0, 0.0, 8)
('Consequential_Oversimplification', 0.0, 0.0, 0.0, 8)
('Repetition', 0.0, 0.0, 0.0, 10)
('Obfuscation-Vagueness-Confusion', 0.0, 0.0, 0.0, 11)
('Appeal_to_Popularity', 0.0, 0.0, 0.0, 22)
('Flag_Waving', 0.0, 0.0, 0.0, 28)
('Guilt_by_Association', 0.21739130434782608, 0.3125, 0.16666666666666666, 30)
('Appeal_to_Fear-Prejudice', 0.3703703703703704, 0.5555555555555556, 0.2777777777777778, 36)
('Appeal_to_Authority', 0.0, 0.0, 0.0, 40)
('Conversation_Killer', 0.0, 0.0, 0.0, 40)
('Exaggeration-Minimisation', 0.04347826086956522, 0.16666666666666666, 0.025, 40)
('Appeal_to_Values', 0.10169491525423728, 0.3333333333333333, 0.06, 50)
('Questioning_the_Reputation', 0.03278688524590164, 0.25, 0.017543859649122806, 57)
('Appeal_to_Hypocrisy', 0.0, 0.0, 0.0, 76)
('Loaded_Language', 0.23529411764705882, 0.3, 0.1935483870967742, 93)
('Doubt', 0.37623762376237624, 0.3584905660377358, 0.3958333333333333, 96)
('Name_Calling-Labeling', 0.46640316205533594, 0.4154929577464789, 0.5315315315315315, 111)
```

## Russian

micro-F1=0.40393       macro-F1=0.11573
```
('Red_Herring', 0.0, 0.0, 0.0, 1)
('Appeal_to_Time', 0.0, 0.0, 0.0, 1)
('Appeal_to_Authority', 0.0, 0.0, 0.0, 2)
('Appeal_to_Popularity', 0.0, 0.0, 0.0, 2)
('Whataboutism', 0.0, 0.0, 0.0, 4)
('Causal_Oversimplification', 0.0, 0.0, 0.0, 6)
('Guilt_by_Association', 0.36363636363636365, 0.5, 0.2857142857142857, 7)
('Appeal_to_Values', 0.11764705882352941, 0.1111111111111111, 0.125, 8)
('Straw_Man', 0.0, 0.0, 0.0, 9)
('Flag_Waving', 0.0, 0.0, 0.0, 10)
('Obfuscation-Vagueness-Confusion', 0.0, 0.0, 0.0, 10)
('False_Dilemma-No_Choice', 0.0, 0.0, 0.0, 11)
('Slogans', 0.0, 0.0, 0.0, 11)
('Appeal_to_Fear-Prejudice', 0.0, 0.0, 0.0, 13)
('Consequential_Oversimplification', 0.0, 0.0, 0.0, 13)
('Appeal_to_Hypocrisy', 0.0, 0.0, 0.0, 17)
('Repetition', 0.0, 0.0, 0.0, 20)
('Conversation_Killer', 0.14634146341463414, 0.17647058823529413, 0.125, 24)
('Exaggeration-Minimisation', 0.20512820512820512, 0.3333333333333333, 0.14814814814814814, 27)
('Name_Calling-Labeling', 0.3857142857142857, 0.27835051546391754, 0.627906976744186, 43)
('Questioning_the_Reputation', 0.43689320388349523, 0.4017857142857143, 0.4787234042553192, 94)
('Doubt', 0.4327485380116958, 0.3148936170212766, 0.6915887850467289, 107)
('Loaded_Language', 0.5736434108527131, 0.46835443037974683, 0.74, 150)
```