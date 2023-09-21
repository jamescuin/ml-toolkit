###################### Imports #################################

import logging
import os
import pandas as pd
import torch
from lib.plotting.plotting_utils import visualise_training_performance
from lib.utils import log_constants, report_gpu, save_configurations, save_test_results_to_json
import configs.downstream.base_config as base_config
import configs.downstream.variable_config as variable_config
from lib.data.data_getter import DataGetter
from lib.data.downstream.data_splitting import get_datasplits_kfold, tokenize_datasplits_kfold, get_dataloaders_kfold
from lib.models.handlers.model_loader import ModelLoader
from lib.models.training.downstream.training_loop_sentiment import train_model_sentiment_kfold
from lib.models.testing.downstream.testing_loop_sentiment import test_model_sentiment_kfold, calculate_test_results_across_folds
from lib.data.data_caching import cache_dict_as_pt, load_dict_from_pt
from torch.cuda import OutOfMemoryError
from lib.models.model_utils import get_token_lengths_nth_percentile
from transformers import set_seed


###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)

# TODO: Find out what below is for.  
os.environ['TRANSFORMERS_CACHE'] = '/data/ds-eu-west-2-efs/hf_cache'
os.environ["CUDA_VISIBLE_DEVICES"]= "0"


###################### Functions #################################

def main():
    log.info('Starting Downstream Sentiment Analysis Task...')

    # Set seed before initializing model.
    set_seed(42)

    log.info('Reading in constants from config files...')
    DOWNSTREAM_ALIAS = base_config.DOWNSTREAM_ALIAS
    PRETRAINING_MODEL_TYPE = base_config.PRETRAINING_MODEL_TYPE
    
    DOWNSTREAM_VARIABLES = variable_config.DOWNSTREAM_VARIABLES
    FILEPATHS = variable_config.FILEPATHS
    LOGGING_FILEPATHS = variable_config.LOGGING_FILEPATHS
    ADAPTER_CONFIG = variable_config.ADAPTER_CONFIG
    MODEL_OPTIMIZATION_CONFIG = variable_config.MODEL_OPTIMIZATION_CONFIG
    DATA_CACHING = variable_config.DATA_CACHING

    log.info('Saving copy of configurations...')
    save_configurations(
        [FILEPATHS, LOGGING_FILEPATHS, DOWNSTREAM_VARIABLES, MODEL_OPTIMIZATION_CONFIG, ADAPTER_CONFIG, DATA_CACHING],
        f"{LOGGING_FILEPATHS['CONFIGURATION_DIRECTORY_PATH']}/{DOWNSTREAM_ALIAS}.json",
        info=True
    )

    log.info(f'Using Pretraining Alias: {DOWNSTREAM_ALIAS}')

    log.info('Logging configurations...')
    for config in [FILEPATHS, DOWNSTREAM_VARIABLES]:
        log_constants(config)

    log.info('Reporting on GPU and clearing cache...')
    report_gpu()

    log.info("Attempting to assign device to GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======================= Raw Data Retrieval =======================
    log.info('Retrieving data for Downstream Sentiment Analysis...')
    # sentiment_df = pd.read_csv('/data/ds-eu-west-2-efs/james/external_data/kaggle/kaggle_sentiment_data_v2_fixed.csv')
    sentiment_df = pd.read_csv('/data/ds-eu-west-2-efs/james/external_data/SEntFIN/SEntFIN_data_fixed.csv')

    # ======================= Tokenizer Loading =======================
    model_loader = ModelLoader(
        model_path=FILEPATHS['PRETRAINED_MODEL_DIRECTORY_PATH'],
        model_type=PRETRAINING_MODEL_TYPE,
        device=device,
        )
    tokenizer = model_loader.load_tokenizer()

    log.info('Getting max_token_length...')
    MAX_INPUT_LENGTH = get_token_lengths_nth_percentile(df=sentiment_df, tokenizer=tokenizer, n=99)  + len(tokenizer.batch_encode_plus(["multi-class classification: "]))
    MAX_TARGET_LENGTH = 2
    log.info(f'MAX_INPUT_LENGTH: {MAX_INPUT_LENGTH}, MAX_TARGET_LENGTH: {MAX_TARGET_LENGTH}')

    # ======================= Tokenized Data Retrieval =======================
    if DATA_CACHING['USE_TOKENIZED_DATASPLITS_CACHE']:
        log.info('Using cached tokenized datasplits...')
        tokenized_datasplits_kfold = load_dict_from_pt(DATA_CACHING['TOKENIZED_DATASPLITS_CACHE_PATH'])
        log.info(f"Loaded cached tokenized datasplits from: {DATA_CACHING['TOKENIZED_DATASPLITS_CACHE_PATH']}")
    else:
        log.info('Split data into train, validation, and test data, for each fold...')
        datasplits_kfold = get_datasplits_kfold(
            df=sentiment_df,
            target_col='Sentiment',
            num_folds=DOWNSTREAM_VARIABLES['NUM_FOLDS'],
            stratify=DOWNSTREAM_VARIABLES['STRATIFY'],
            info=False,
        )
        log.info('Tokenizing datasplits...')
        tokenized_datasplits_kfold = tokenize_datasplits_kfold(
            datasplits_kfold=datasplits_kfold, 
            tokenizer=tokenizer, 
            max_input_length=MAX_INPUT_LENGTH, 
            max_target_length=MAX_TARGET_LENGTH,
            input_col='Sentence',
            target_col='Sentiment',
        )
        if DATA_CACHING['CACHE_TOKENIZED_DATASPLITS']:
            log.info('Caching tokenized datasplits....')
            cache_dict_as_pt(tokenized_datasplits_kfold, DATA_CACHING['TOKENIZED_DATASPLITS_CACHE_PATH'])
            log.info(f"Cached tokenized datasplits at: {DATA_CACHING['TOKENIZED_DATASPLITS_CACHE_PATH']}")

    # ======================= Training Loop =======================
    while DOWNSTREAM_VARIABLES['BATCH_SIZE'] > 0:
        try:
            log.info('Create respective DataLoaders...')
            dataloaders_kfold = get_dataloaders_kfold(
                tokenized_datasplits_kfold,
                batch_size=DOWNSTREAM_VARIABLES['BATCH_SIZE'],
            )
            log.info('Begin training loop, for each fold...')
            training_results = train_model_sentiment_kfold(
                dataloaders_kfold=dataloaders_kfold,
                pretrained_model_directory_path=FILEPATHS['PRETRAINED_MODEL_DIRECTORY_PATH'], 
                checkpoint_directory_path=FILEPATHS['CHECKPOINT_DIRECTORY_PATH'], 
                best_model_directory_path=FILEPATHS['DOWNSTREAM_MODEL_DIRECTORY_PATH'],
                pretrained_model_type=PRETRAINING_MODEL_TYPE,
                tensorboard_dir=LOGGING_FILEPATHS['TENSORBOARD_DIRECTORY_PATH'],
                device=device,
                adapter_type=ADAPTER_CONFIG['ADAPTER_TYPE'], 
                add_adapter=ADAPTER_CONFIG['ADD_ADAPTER'],
                optimizer_type=MODEL_OPTIMIZATION_CONFIG['OPTIMIZER_TYPE'],
                scheduler_type=MODEL_OPTIMIZATION_CONFIG['SCHEDULER_TYPE'],
                learning_rate=MODEL_OPTIMIZATION_CONFIG['LEARNING_RATE'],
                weight_decay=MODEL_OPTIMIZATION_CONFIG['WEIGHT_DECAY'],
                num_warmup_steps=MODEL_OPTIMIZATION_CONFIG['NUM_WARMUP_STEPS'],
                num_cycles=DOWNSTREAM_VARIABLES['NUM_CYCLES'],
                use_latest_checkpoint=DOWNSTREAM_VARIABLES['USE_LATEST_MODEL_CHECKPOINT'],
                checkpoint_save_mode=DOWNSTREAM_VARIABLES['CHECKPOINTING_SAVE_MODE'],
                show_progress_bars=True,
                epochs_per_cycle=DOWNSTREAM_VARIABLES['EPOCHS_PER_CYCLE'],
                max_target_length=MAX_TARGET_LENGTH,
                show_n_predictions=10,
                train_n_folds='all',
            )

            log.info('Training loop ended!')

            break
        # Iteratively reduce batch size from original PRETRAINING_VARIABLES['BATCH_SIZE'] value to avoid OutOfMemoryError.
        except OutOfMemoryError as e:  # noqa 
            log.info(e)
            log.info('Reducing Batch Size...')
            DOWNSTREAM_VARIABLES['BATCH_SIZE'] -= 10
            log.info(f"New Batch Size: {DOWNSTREAM_VARIABLES['BATCH_SIZE']}")

            log.info('Saving copy of updated configurations...')
            save_configurations(
                [FILEPATHS, LOGGING_FILEPATHS, DOWNSTREAM_VARIABLES, MODEL_OPTIMIZATION_CONFIG, ADAPTER_CONFIG, DATA_CACHING],
                f"{LOGGING_FILEPATHS['CONFIGURATION_DIRECTORY_PATH']}/{DOWNSTREAM_ALIAS}.json",
                info=True
            )
            log.info('Reporting on GPU and clearing cache...')
            report_gpu()

    # ======================= Plotting Training Results =======================
    for fold, fold_training_results in training_results.items():
        log.info(f'Visualising model performance over epochs for {fold}...')
        visualise_training_performance(
                train_tracker=fold_training_results['train_tracker'],
                val_tracker=fold_training_results['val_tracker'],
                plot_directory_path=f"{LOGGING_FILEPATHS['PLOT_DIRECTORY_PATH']}/{fold}",
                plot_filename='training_val_training_performance.png',
                metric='f1',
            )
    
    log.info('Reporting on GPU and clearing cache...')
    report_gpu()

    # ======================= Testing Loop =======================
    log.info('Begin testing model on Test Data, for each fold...')
    test_results_per_fold = test_model_sentiment_kfold(
        dataloaders_kfold=dataloaders_kfold,
        pretrained_model_type='t5_flax',
        pretrained_model_dir=FILEPATHS['PRETRAINED_MODEL_DIRECTORY_PATH'],
        best_model_directory_path=FILEPATHS['DOWNSTREAM_MODEL_DIRECTORY_PATH'],
        device=device,
        adapter_type=ADAPTER_CONFIG['ADAPTER_TYPE'],
        show_n_predictions=10,
        show_progress_bars=True,
        max_target_length=MAX_TARGET_LENGTH,
        using_adapter=ADAPTER_CONFIG['ADD_ADAPTER'],
    )

    log.info('Calcuating average test results across folds...')
    test_results = calculate_test_results_across_folds(
        test_results_per_fold=test_results_per_fold
    )

    results_to_save = {
        'Avg Acc': [test_results[0]], 
        'Avg Prec': [test_results[1]], 
        'Avg Recall': [test_results[2]], 
        'Avg F1': [test_results[3]],
        'Std Acc': [test_results[4]], 
        'Std Prec': [test_results[5]], 
        'Std Recall': [test_results[6]], 
        'Std F1': [test_results[7]],
        }
    
    save_test_results_to_json(
        results_to_save, 
        f"{LOGGING_FILEPATHS['TEST_RESULTS_DIRECTORY_PATH']}/test_results.json", 
        info=True
    )
    log.info('Testing ended!')


    log.info('Finished Downstream Task!')

    pass

    

if __name__ == "__main__":
    main()
