local is_cose = if std.findSubstr('cose', std.extVar('TRAIN_DATA_PATH')) == [] then false else true;

{
  dataset_reader : {
    type : "rationale_reader" + (if is_cose then '_cose' else ''),
    token_indexers : {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
    },
  },
  validation_dataset_reader: {
    type : "rationale_reader" + (if is_cose then '_cose' else ''),
    token_indexers : {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
    },
  },
  train_data_path: std.extVar('TRAIN_DATA_PATH'),
  validation_data_path: std.extVar('DEV_DATA_PATH'),
  test_data_path: std.extVar('TEST_DATA_PATH'),
  model: {
    type: "soft_encoder_rationale_model" + (if is_cose then '_cose' else ''),
    text_field_embedder: {
      token_embedders: {
        tokens: {
          "type": "embedding",
          "pretrained_file": "http://nlp.stanford.edu/data/glove.42B.300d.zip",
          "embedding_dim": 300,
          "trainable": true
        },
      },
    },
    seq2seq_encoder : {
      type: 'lstm',
      input_size: 300,
      hidden_size: 128,
      num_layers: 1,
      bidirectional: true
    },
    dropout: 0.2,
    attention: {
      type: 'additive',
      vector_dim: 256,
      matrix_dim: 256,
    },
    feedforward_encoder:{
      input_dim: 256,
      num_layers: 1,
      hidden_dims: [128],
      activations: ['relu'],
      dropout: 0.2
    },
    saliency_scorer : {
      type: 'simple_gradient',
      threshold: std.extVar('threshold'),
      batch_size: -1
    },
  },
  iterator: {
    type: "bucket",
    sorting_keys: [['document', 'num_tokens']],
    batch_size : std.extVar("batch_size"),
    biggest_batch_first: true
  },
  trainer: {
    num_epochs: 20,
    patience: 5,
    grad_norm: 10.0,
    validation_metric: "+accuracy",
    num_serialized_models_to_keep: 1,
    cuda_device: std.extVar("CUDA_DEVICE"),
    optimizer: {
      type: "adam",
      lr: 1e-2
    }
  },
  random_seed:  std.parseInt(std.extVar("SEED")),
  pytorch_seed: std.parseInt(std.extVar("SEED")),
  numpy_seed: std.parseInt(std.extVar("SEED")),
  evaluate_on_test: true
}
