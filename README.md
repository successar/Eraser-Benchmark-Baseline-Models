For Lei et al. Encoder Generator Model
---------------------------------------

1. For training a bert encoder generator model :
```bash
dataset_folder=data/movies \
dataset_name=movies \
classifier=bert_encoder_generator \
output_dir=outputs \
exp_name=$EXP_NAME \
batch_size=4 \
rs_weight="use 1 if using rationale supervision else 0" \
bash Rationale_model/commands/model_train_script.sh
```

2. For making prediction on test set :
```bash
dataset_folder=data/movies \
dataset_name=movies \
classifier=bert_encoder_generator \
output_dir=outputs \
exp_name=$EXP_NAME \
batch_size=4 \
bash Rationale_model/commands/model_predict.sh
```

2. For calculating metrics : 
```bash
python rationale_benchmark/metrics.py \
--data_dir data/movies \
--split test \
--results outputs/bert_encoder_generator/movies/$EXP_NAME/test_prediction.jsonl
--score_file outputs/bert_encoder_generator/movies/$EXP_NAME/test_scores.json
```

For Soft Scores BERT-LSTM Model
---------------------

1. For training a bert soft scores model :
```bash
dataset_folder=data/movies \
dataset_name=movies \
classifier=soft_bert \
output_dir=outputs \
exp_name=$EXP_NAME \
batch_size=4 \
threshold=0.0 \
saliency=wrapper \
bash Rationale_model/commands/model_train_script.sh
```
2. For making prediction using saliency method (For attention, use $saliency below with `wrapper` and for simple gradient, replace it with `simple_gradient`). For computing comprehensiveness and sufficiency metrics, use the threshold provided in paper as a number in (0, 1), not percentages.

```bash
dataset_folder=data/movies \
dataset_name=movies \
classifier=soft_bert \
output_dir=outputs \
exp_name=$EXP_NAME \
batch_size=4 \
saliency=$saliency \
threshold=$threshold \
bash Rationale_model/commands/model_predict.sh
```

3. For calculating metrics : 

```bash
python rationale_benchmark/metrics.py \
--data_dir data/movies \
--split test \
--results outputs/soft_bert/movies/$EXP_NAME/$saliency/test_prediction.jsonl
--score_file outputs/soft_bert/movies/$EXP_NAME/$saliency/test_scores.json
```
To train GloVe-LSTM models, please replace `soft_bert` with `soft_word_emb` in above code.