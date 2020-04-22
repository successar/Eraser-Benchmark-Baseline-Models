export data_base_path=${dataset_folder:?"set dataset folder"}

export archive=$output_dir/${classifier:?"set classifier"}/${dataset_name:?"enter dataset name"}/${exp_name:?"set exp name"}
export TEST_DATA_PATH=$data_base_path/test.jsonl
export VAL_DATA_PATH=$data_base_path/val.jsonl

mkdir -p $archive/$saliency

export batch_size_saliency=${batch_size_saliency-1} 

echo "cuda device is ${CUDA_DEVICE}"

# allennlp predict \
# --output-file $archive/$saliency/val_prediction.jsonl \
# --predictor rationale_predictor \
# --include-package Rationale_model \
# --silent \
# --cuda-device $CUDA_DEVICE \
# --batch-size $batch_size \
# -o "{model: {saliency_scorer: {type: \"$saliency\", threshold: $threshold, batch_size: $batch_size_saliency}}}" \
# --use-dataset-reader \
# --dataset-reader-choice validation \
# $archive/model.tar.gz $VAL_DATA_PATH

allennlp predict \
--output-file $archive/$saliency/test_prediction${seed}.jsonl \
--predictor rationale_predictor \
--include-package Rationale_model \
--silent \
--cuda-device $CUDA_DEVICE \
--batch-size $batch_size \
-o "{model: {saliency_scorer: {type: \"$saliency\", threshold: $threshold, batch_size: $batch_size_saliency}}}" \
--use-dataset-reader \
--dataset-reader-choice validation \
$archive/model.tar.gz $TEST_DATA_PATH