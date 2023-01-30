## TODOs

Deadline: submission deadline is 1/30

Our focus is on subtask 3: persuasion techniques


TODOs 1/30:
 - first thing: run on test set and make a submission
 - figure out how many eps is the best
    - (screen) ge: en, fr:it, 6 eps each
    - po: 11 eps
 - It's sus how train size is much smaller than val and dev, investigate (ex. 235851 11591 227952 75048)
    - for your final submission you might want to train on everything
 - isolate the evaluation function and test polish dev set on polish to english translation

TODOs 1/29:
 - skip pretrain on future models because pretrain now takes more than 3 hours

TODOs 1/28: 
 - first make sure that your condensed training script can actually run
 - clean up the googletrans files, make sure for EACH language we have a one-to-one correspondence between template and label files
 - pretrain on the googletrans file
 - finetune on the uncommon labels
 - maybe translate to the surprise languages (es, gr, ka) as well?

TODOs as of 1/19:
 - train on english dataset with 2021 task 6 data
 - restructure the code and make it modular
 - how do we make the model perform well on the other languages?
 - start summarizing our method and draft the paper  