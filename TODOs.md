## TODOs

Deadline: submission deadline is 1/30

Our focus is on subtask 3: persuasion techniques

TODOs update 1/22
 - Augment every language using EVERY other language

```python
languages_list = ["en", "fr", "ge", "it", "po", "ru"]
for lang_1 in languages_list:
    for lang_2 in languages_list:
        if lang_1 != lang_2:
            translate_1_to_2(src=lang_1, target=lang_2) # translate lang1 to lang2
```

 - Should also translate the label definitions to all 6 languages

TODOs as of 1/19:
 - train on english dataset with 2021 task 6 data
 - restructure the code and make it modular
 - how do we make the model perform well on the other languages?
 - start summarizing our method and draft the paper  