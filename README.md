# SemEval23-Task-3-UIUC-Team

This repo contains the source code for our SemEval 2023 workshop paper - 

Team NLUBot101 at SemEval-2023 Task 3: An Augmented Multilingual NLI Approach Towards Online News Persuasion Techniques Detection

## Usage

```
# train the model:
python src/subtask_3_multiling.py False

# Evaluation only:
python src/subtask_3_multiling.py True

# run scorer:
python scorers/scorer-subtask-3.py \
    -p results/submission/googletrans-TEST-output-subtask3-en_def.txt 
    -g data/en/dev-labels-subtask-3.txt \
    --techniques_file_path scorers/techniques_subtask3.txt

```
