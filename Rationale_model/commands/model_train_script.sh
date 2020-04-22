export data_base_path=${dataset_folder:?"set dataset folder"}

export CONFIG_FILE=Rationale_model/training_config/classifiers/${classifier:?"set classifier"}.jsonnet

export TRAIN_DATA_PATH=${data_base_path:?"set data base path"}/train.jsonl
export DEV_DATA_PATH=$data_base_path/val.jsonl
export TEST_DATA_PATH=$data_base_path/test.jsonl

export OUTPUT_BASE_PATH=${output_dir:-outputs}/$classifier/${dataset_name:?"set dataset name"}/${exp_name:?"set exp name"}

export SEED=${random_seed:-100}

allennlp train -s $OUTPUT_BASE_PATH --include-package Rationale_model --force $CONFIG_FILE