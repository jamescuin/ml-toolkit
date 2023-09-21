import configs.downstream.base_config as base_config

FILEPATHS = {
    # 'PRETRAINED_MODEL_DIRECTORY_PATH': f'/data/ds-eu-west-2-efs/james/pretraining-james/pretraining/model_checkpoints/{base_config.DOWNSTREAM_ALIAS}/best_model', # Path to load the pretrained model from.
    # 'PRETRAINED_MODEL_DIRECTORY_PATH': "jamescuin/tpu-pretraining-james-020", # (HuggingFace Option)
    'PRETRAINED_MODEL_DIRECTORY_PATH': 'google/flan-t5-base', # (HuggingFace Option)
    'DOWNSTREAM_MODEL_DIRECTORY_PATH': f'/data/ds-eu-west-2-efs/james/pretraining-james/downstream/model_checkpoints/{base_config.DOWNSTREAM_ALIAS}/best_model', # Path to save models trained on sentiment analysis task
    'CHECKPOINT_DIRECTORY_PATH': f'/data/ds-eu-west-2-efs/james/pretraining-james/downstream/model_checkpoints/{base_config.DOWNSTREAM_ALIAS}', # Path to save a checkpoint of the model to, after each training epoch. 
}

LOGGING_FILEPATHS = {
    'CONFIGURATION_DIRECTORY_PATH': f'/home/james/git_repos/pretraining-james/logs/configs/downstream/{base_config.DOWNSTREAM_ALIAS}',
    'TENSORBOARD_DIRECTORY_PATH': f'/home/james/git_repos/pretraining-james/logs/tensorboard_logs/downstream/{base_config.DOWNSTREAM_ALIAS}',
    'PLOT_DIRECTORY_PATH': f'/home/james/git_repos/pretraining-james/logs/plots/downstream/{base_config.DOWNSTREAM_ALIAS}',
    'TEST_RESULTS_DIRECTORY_PATH': f'/home/james/git_repos/pretraining-james/logs/test_results/downstream/{base_config.DOWNSTREAM_ALIAS}',
}

DATA_PATHS = {
    # 'kaggle': '/data/ds-eu-west-2-efs/guilherme/jamie_data/kaggle_sentiment_data.csv', # Path to Kaggle data provided by Guilherme.
    'kaggle': '/data/ds-eu-west-2-efs/james/external_data/kaggle/kaggle_sentiment_data_v2_fixed.csv', # Path to Kaggle data (I fixed)
    'cnn_dailymail_dutch': 'yhavinga/cnn_dailymail_dutch',
}

DOWNSTREAM_VARIABLES = {
    'DATA_SOURCE': {
        'rows': None, 
        'path': DATA_PATHS[base_config.DATA_SOURCE], 
        'source': base_config.DATA_SOURCE, 
        'cache_dir': f'/data/ds-eu-west-2-efs/james/pretraining-james/cache/data/{base_config.DATA_SOURCE}'
    },
    'BATCH_SIZE': 128,
    'NUM_CYCLES': 10,
    'EPOCHS_PER_CYCLE': 1,
    'NUM_FOLDS': 1,
    'STRATIFY': True,
    'CHECKPOINTING_SAVE_MODE': 'overwrite', # The way in which models checkpoints are saved. 
    'USE_LATEST_MODEL_CHECKPOINT': False, # Whether we wish to use the latest model checkpoint as start point for training
}

MODEL_OPTIMIZATION_CONFIG = {
    'OPTIMIZER_TYPE': 'adam_w',
    'SCHEDULER_TYPE': 'linear',
    'LEARNING_RATE': 1e-3,
    'WEIGHT_DECAY': 1e-3,
    'NUM_WARMUP_STEPS': 0,
}

ADAPTER_CONFIG = {
    'ADD_ADAPTER': True,
    'ADAPTER_TYPE': 'lora',
}

DATA_CACHING = {
    'CACHE_TOKENIZED_DATASPLITS': True,
    'USE_TOKENIZED_DATASPLITS_CACHE': False,
    'TOKENIZED_DATASPLITS_CACHE_PATH': f'/data/ds-eu-west-2-efs/james/pretraining-james/cache/data/{base_config.DATA_SOURCE}/tokenized_dataplits_{DOWNSTREAM_VARIABLES["DATA_SOURCE"]["rows"]}_rows_{DOWNSTREAM_VARIABLES["NUM_FOLDS"]}_folds_cache.pt'
}