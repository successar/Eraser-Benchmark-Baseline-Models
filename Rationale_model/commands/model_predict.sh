export data_base_path=${dataset_folder:?"set dataset folder"}
export TEST_DATA_PATH=$data_base_path/test.jsonl

export archive=${output_dir:?"Set output_dir"}/${classifier:?"set classifier"}/${dataset_name:?"enter dataset name"}/${exp_name:?"set exp name"}

allennlp predict \
--output-file $archive/test_prediction.jsonl \
--predictor rationale_predictor \
--include-package Rationale_model \
--silent \
--cuda-device $CUDA_DEVICE \
--batch-size $batch_size \
--use-dataset-reader \
--dataset-reader-choice validation \
$archive/model.tar.gz $TEST_DATA_PATH