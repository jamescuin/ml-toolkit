import configs.pretraining.base_config as base_config

FILEPATHS = {
    'PRETRAINED_MODEL_DIRECTORY_PATH': f'/data/ds-eu-west-2-efs/james/pretraining-james/pretraining/model_checkpoints/{base_config.PRETRAINING_ALIAS}/best_model', # Path to directory to save the best perfroming model to.
    'CHECKPOINT_DIRECTORY_PATH': f'/data/ds-eu-west-2-efs/james/pretraining-james/pretraining/model_checkpoints/{base_config.PRETRAINING_ALIAS}', # Path to save a checkpoint of the model to, after each training epoch. 
}

LOGGING_FILEPATHS = {
    'CONFIGURATION_DIRECTORY_PATH': f'/home/james/git_repos/pretraining-james/logs/configs/pretraining/{base_config.PRETRAINING_ALIAS}',
    'TENSORBOARD_DIRECTORY_PATH': f'/home/james/git_repos/pretraining-james/logs/tensorboard_logs/pretraining/{base_config.PRETRAINING_ALIAS}',
    'PLOT_DIRECTORY_PATH': f'/home/james/git_repos/pretraining-james/logs/plots/pretraining/{base_config.PRETRAINING_ALIAS}',
    'TEST_RESULTS_DIRECTORY_PATH': f'/home/james/git_repos/pretraining-james/logs/test_results/pretraining/{base_config.PRETRAINING_ALIAS}',
}

DATA_PATHS = {
    'reuters': '/data/ds-eu-west-2-efs/guilherme/external_data/reuters/reuters', # Path to Reuters data directory.
    'bloomberg': '/data/ds-eu-west-2-efs/james/external_data/bloomberg/bloomberg_20061020_20131126_relevant_data.parquet', # Path to Bloomberg .parquet file.
    'mc4_nl_cleaned': 'yhavinga/mc4_nl_cleaned',
}

PRETRAINING_VARIABLES = {
    'DATA_SOURCE': {
        'rows': 1000, 
        'path': DATA_PATHS[base_config.DATA_SOURCE], 
        'source': base_config.DATA_SOURCE,
        'cache_dir': f'/data/ds-eu-west-2-efs/james/pretraining-james/cache/data/{base_config.DATA_SOURCE}'
        },
    'MAX_LENGTH': None,  # The maximum total input sequence length after tokenization and masking. Sequences longer than this will be truncated. If None, default to the max input length of the model.
    'MLM_PROBABILITY': 0.15, # Ratio of tokens to mask for span masked language modeling loss
    'MEAN_NOISE_SPAN_LENGTH': 3, # Mean span length of masked tokens
    'BATCH_SIZE': 10,
    'NUM_CYCLES': 1, # This might need to be higher for the model to get good -- debug small and then go up
    'EPOCHS_PER_CYCLE': 3,
    'CHECKPOINTING_SAVE_MODE': 'overwrite', # The way in which models checkpoints are saved. 
    'USE_LATEST_MODEL_CHECKPOINT': False, # Whether we wish to use the latest model checkpoint as start point for training
}

MODEL_OPTIMIZATION_CONFIG = {
    'OPTIMIZER_TYPE': 'adam_w',
    'SCHEDULER_TYPE': 'linear',
    'LEARNING_RATE': 5e-5,  
    'WEIGHT_DECAY': 0.01,  
    'NUM_WARMUP_STEPS': 0,  
}
    
ADAPTER_CONFIG = {
    'ADD_ADAPTER': False,
    'MERGE_ADAPTER': False,
    'ADAPTER_TYPE': 'lora',
}

COLLATOR_CONFIG = {
    'ACTIVE_COLLATOR': 'DataCollatorForT5MLMBloomberg',  # Options are: DataCollatorForT5MLMReuters, DataCollatorForT5MLMBloomberg, DataCollatorForUL2Bloomberg
    'DataCollatorForT5MLMReuters': {},
    'DataCollatorForT5MLMBloomberg': {}, 
    'DataCollatorForT5UL2Bloomberg': {
        'R_DENOISING': True,
        'R_PROB': 3/7,
        'R_CONFIG': ((3, 0.15), (8, 0.15), ),
        'S_DENOISING': True,
        'S_PROB': 1/7,
        'X_DENOISING': True,
        'X_PROB': 4/7,
        'X_CONFIG': ((3, 0.5), (8, 0.5), (64, 0.15), (64, 0.5), ),
    },
}

DATA_CACHING = {
    'CACHE_TOKENIZED_DATASPLITS': True,
    'USE_TOKENIZED_DATASPLITS_CACHE': True,
    'TOKENIZED_DATASPLITS_CACHE_PATH': f'/data/ds-eu-west-2-efs/james/pretraining-james/cache/data/{base_config.DATA_SOURCE}/tokenized_dataplits_{PRETRAINING_VARIABLES["DATA_SOURCE"]["rows"]}_rows_{PRETRAINING_VARIABLES["MAX_LENGTH"]}_ml_{PRETRAINING_VARIABLES["MLM_PROBABILITY"]}_mlmp_{PRETRAINING_VARIABLES["MEAN_NOISE_SPAN_LENGTH"]}_mnsl_cache.pt'
}