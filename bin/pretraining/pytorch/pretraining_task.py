###################### Imports #################################

from lib.data.data_caching import cache_dict_as_pt, load_dict_from_pt
from lib.models.handlers.model_optimization import ModelOptimization
from lib.models.handlers.adapter_handler import AdapterHandler
from lib.data.data_getter import DataGetter
from lib.models.handlers.model_loader import ModelLoader
import configs.pretraining.preprocessing_config as preprocessing_config
import configs.pretraining.variable_config as variable_config
import configs.pretraining.base_config as base_config
from torch.cuda import OutOfMemoryError
from lib.utils import log_constants, report_gpu, save_configurations, convert_function_values_to_name, save_test_results_to_json
from lib.plotting.plotting_utils import visualise_training_performance
from lib.data.pretraining.data_splitting import get_datasplits, get_dataloaders, get_tokenized_datasplits
from lib.data.pretraining.data_collators import get_data_collator
from lib.models.training.pretraining.training_loop import train_model
from lib.models.testing.pretraining.testing_loop import test_model
from lib.models.model_utils import compute_input_and_target_lengths
import lib.data.preprocessing_utils as ppu
import torch
import pandas as pd
import logging
import os


###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)

# TODO: Find out what below is for.
os.environ['TRANSFORMERS_CACHE'] = '/data/ds-eu-west-2-efs/hf_cache'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


###################### Functions #################################

def main():
    log.info('Starting Pre-Training...')

    log.info('Reading in variables from config files...')
    BASE_MODEL = base_config.BASE_MODEL
    BASE_MODEL_TYPE = base_config.BASE_MODEL_TYPE
    PRETRAINING_ALIAS = base_config.PRETRAINING_ALIAS
    DATA_SOURCE = base_config.DATA_SOURCE

    PRETRAINING_VARIABLES = variable_config.PRETRAINING_VARIABLES
    FILEPATHS = variable_config.FILEPATHS
    LOGGING_FILEPATHS = variable_config.LOGGING_FILEPATHS
    ADAPTER_CONFIG = variable_config.ADAPTER_CONFIG
    MODEL_OPTIMIZATION_CONFIG = variable_config.MODEL_OPTIMIZATION_CONFIG
    PREPROCESSING_CONFIG = preprocessing_config.PREPROCESSING_CONFIG
    COLLATOR_CONFIG = variable_config.COLLATOR_CONFIG
    DATA_CACHING = variable_config.DATA_CACHING

    PREPROCESSING_CONFIG_FOR_LOGS = convert_function_values_to_name(PREPROCESSING_CONFIG)

    log.info(f'Using Base Model: {BASE_MODEL}')
    log.info(f'Using Pretraining Alias: {PRETRAINING_ALIAS}')

    log.info('Logging Defined Filepaths...')
    log_constants(FILEPATHS)

    log.info('Logging Defined Pretraining Variables...')
    log_constants(PRETRAINING_VARIABLES)

    log.info('Logging Pre-Processing Config...')
    log_constants(PREPROCESSING_CONFIG)

    log.info('Logging Collator Config...')
    log_constants(COLLATOR_CONFIG)

    log.info('Logging Adaptor Config...')
    log_constants(ADAPTER_CONFIG)

    log.info('Logging Model Optimization Config...')
    log_constants(MODEL_OPTIMIZATION_CONFIG)

    log.info('Saving copy of configurations...')
    save_configurations(
        [FILEPATHS, LOGGING_FILEPATHS, PRETRAINING_VARIABLES, MODEL_OPTIMIZATION_CONFIG, ADAPTER_CONFIG, DATA_CACHING, PREPROCESSING_CONFIG_FOR_LOGS, COLLATOR_CONFIG],
        f"{LOGGING_FILEPATHS['CONFIGURATION_DIRECTORY_PATH']}/pretraining_config.json",
        info=True
    )
    log.info('Reporting on GPU and clearing cache...')
    report_gpu()

    log.info("Attempting to assign device to GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======================= Raw Data Retrieval =======================
    data_source = PRETRAINING_VARIABLES['DATA_SOURCE']['source']
    data_path = PRETRAINING_VARIABLES['DATA_SOURCE']['path']
    data_rows = PRETRAINING_VARIABLES['DATA_SOURCE']['rows']
    cache_dir = PRETRAINING_VARIABLES['DATA_SOURCE']['cache_dir']

    data_getter = DataGetter(
        source=data_source,
        path=data_path,
        rows=data_rows,
        verbose=True
    )
    data_df = data_getter.get_data()

    log.info('Apply Pre-Processing to data...')
    data_df = ppu.apply_preprocessing_functions(data_df, PREPROCESSING_CONFIG)

    # ======================= Model & Tokenizer Loading =======================
    model_loader = ModelLoader(
        model_path=BASE_MODEL,
        model_type=BASE_MODEL_TYPE,
        device=device
    )
    tokenizer = model_loader.load_tokenizer()
    log.info(f"Max input for model: {tokenizer.model_max_length}")
    model = model_loader.load_model()
    tokenizer.save_pretrained(FILEPATHS['PRETRAINED_MODEL_DIRECTORY_PATH'])

    # Adding Adapter (if specified)
    if ADAPTER_CONFIG['ADD_ADAPTER']:
        adapter_handler = AdapterHandler(
            model=model,
            adapter_type=ADAPTER_CONFIG['ADAPTER_TYPE'],
            device=device
        )
        model = adapter_handler.add_adapter_to_model()

    # ======================= Tokenized Data Retrieval =======================
    log.info('Compute required lengths for input and target token sequences...')
    # T5-like span masked language modeling will fuse consecutively masked tokens to a single sentinel token.
    # To ensure that the input length is `max_seq_length`, we need to increase the maximum length
    # according to `mlm_probability` and `mean_noise_span_length`. We can also define the label length accordingly.

    # Calculate minimum of MAX_LENGTH and tokenizer.model_max_length, treating None as -inf.
    max_seq_length = min(filter(None.__ne__, (PRETRAINING_VARIABLES['MAX_LENGTH'], tokenizer.model_max_length)), default=tokenizer.model_max_length)
    log.info(f"max_seq_length: {max_seq_length}")
    EXPANDED_INPUTS_LENGTH, TARGETS_LENGTH = compute_input_and_target_lengths(
        inputs_length=max_seq_length,
        noise_density=PRETRAINING_VARIABLES['MLM_PROBABILITY'],
        mean_noise_span_length=PRETRAINING_VARIABLES['MEAN_NOISE_SPAN_LENGTH'],
    )
    log.info(f'EXPANDED_INPUTS_LENGTH: {EXPANDED_INPUTS_LENGTH}')
    log.info(f'TARGETS_LENGTH: {TARGETS_LENGTH}')

    if DATA_CACHING['USE_TOKENIZED_DATASPLITS_CACHE']:
        log.info('Using cached tokenized datasplits...')
        tokenized_datasplits = load_dict_from_pt(DATA_CACHING['TOKENIZED_DATASPLITS_CACHE_PATH'])
        log.info(f"Loaded cached tokenized datasplits from: {DATA_CACHING['TOKENIZED_DATASPLITS_CACHE_PATH']}")
    else:
        log.info('Split data into train, validation, and test data...')
        datasplits = get_datasplits(
            df=data_df,
            train_split_size=0.8,
            val_split_size=0.1,
            test_split_size=0.1,
        )
        log.info('Tokenizing datasplits...')
        tokenized_datasplits = get_tokenized_datasplits(
            data_source=DATA_SOURCE,
            datasplits=datasplits,
            tokenizer=tokenizer,
            expanded_inputs_length=EXPANDED_INPUTS_LENGTH
        )
        if DATA_CACHING['CACHE_TOKENIZED_DATASPLITS']:
            log.info('Caching tokenized datasplits....')
            cache_dict_as_pt(tokenized_datasplits, DATA_CACHING['TOKENIZED_DATASPLITS_CACHE_PATH'])
            log.info(f"Cached tokenized datasplits at: {DATA_CACHING['TOKENIZED_DATASPLITS_CACHE_PATH']}")

    # ======================= DataCollator Loading =======================
    log.info('Instantiate DataCollatorForT5MLMBloomberg...')
    data_collator = get_data_collator(
        collator_config=COLLATOR_CONFIG,
        tokenizer=tokenizer,
        noise_density=PRETRAINING_VARIABLES['MLM_PROBABILITY'],
        mean_noise_span_length=PRETRAINING_VARIABLES['MEAN_NOISE_SPAN_LENGTH'],
        input_length=max_seq_length,
        target_length=TARGETS_LENGTH
    )

    # ======================= Training Loop =======================
    while PRETRAINING_VARIABLES['BATCH_SIZE'] > 0:
        try:
            log.info('Create respective DataLoaders...')
            dataloaders = get_dataloaders(
                tokenized_datasplits=tokenized_datasplits,
                data_collator=data_collator,
                batch_size=PRETRAINING_VARIABLES['BATCH_SIZE'],
            )

            # Model Optimization
            num_training_steps = PRETRAINING_VARIABLES['NUM_CYCLES'] * len(dataloaders['train'])  # Total number of batches we train on.

            model_optimization = ModelOptimization(
                model=model,
                optimizer_type=MODEL_OPTIMIZATION_CONFIG['OPTIMIZER_TYPE'],
                scheduler_type=MODEL_OPTIMIZATION_CONFIG['SCHEDULER_TYPE']
            )
            optimizer = model_optimization.get_optimizer(
                learning_rate=MODEL_OPTIMIZATION_CONFIG['LEARNING_RATE'],
                weight_decay=MODEL_OPTIMIZATION_CONFIG['WEIGHT_DECAY']
            )
            lr_scheduler = model_optimization.get_scheduler(
                optimizer=optimizer,
                num_warmup_steps=MODEL_OPTIMIZATION_CONFIG['NUM_WARMUP_STEPS'],
                num_training_steps=num_training_steps)

            log.info('Begin training loop...')
            training_results = train_model(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_dataloader=dataloaders['train'],
                val_dataloader=dataloaders['val'],
                device=device,
                num_cycles=PRETRAINING_VARIABLES['NUM_CYCLES'],
                checkpoint_directory_path=FILEPATHS['CHECKPOINT_DIRECTORY_PATH'],
                pretrained_model_directory_path=FILEPATHS['PRETRAINED_MODEL_DIRECTORY_PATH'],
                tensorboard_directory_path=LOGGING_FILEPATHS['TENSORBOARD_DIRECTORY_PATH'],
                use_latest_checkpoint=PRETRAINING_VARIABLES['USE_LATEST_MODEL_CHECKPOINT'],
                checkpoint_save_mode=PRETRAINING_VARIABLES['CHECKPOINTING_SAVE_MODE'],
                show_progress_bars=True,
                epochs_per_cycle=PRETRAINING_VARIABLES['EPOCHS_PER_CYCLE'],
            )

            # Merging Adapter into Model (if specified)
            model = adapter_handler.merge_model(FILEPATHS['PRETRAINED_MODEL_DIRECTORY_PATH'], FILEPATHS['PRETRAINED_MODEL_DIRECTORY_PATH'])

            log.info('Training loop ended!')

            break
        # Iteratively reduce batch size from original PRETRAINING_VARIABLES['BATCH_SIZE'] value to avoid OutOfMemoryError.
        except OutOfMemoryError as e:  # noqa
            log.info(e)
            log.info('Reducing Batch Size...')
            PRETRAINING_VARIABLES['BATCH_SIZE'] -= 2
            log.info(f"New Batch Size: {PRETRAINING_VARIABLES['BATCH_SIZE']}")

            log.info('Saving copy of updated configurations...')
            save_configurations(
                [FILEPATHS, LOGGING_FILEPATHS, PRETRAINING_VARIABLES, MODEL_OPTIMIZATION_CONFIG, ADAPTER_CONFIG, DATA_CACHING, PREPROCESSING_CONFIG_FOR_LOGS, COLLATOR_CONFIG],
                f"{LOGGING_FILEPATHS['CONFIGURATION_DIRECTORY_PATH']}/{PRETRAINING_ALIAS}.json",
                info=True
            )
            log.info('Reporting on GPU and clearing cache...')
            report_gpu()

    # ======================= Plotting Training Results =======================
    log.info('Visualising model performance over epochs...')
    visualise_training_performance(
        train_tracker=training_results['train_tracker'],
        val_tracker=training_results['val_tracker'],
        plot_directory_path=LOGGING_FILEPATHS['PLOT_DIRECTORY_PATH'],
        plot_filename='training_val_loss.png',
        metric='loss',
    )

    log.info('Reporting on GPU and clearing cache...')
    report_gpu()

    # ======================= Testing Loop =======================
    log.info('Begin testing model on Test Data...')
    test_results = test_model(
        model=model,
        tokenizer=tokenizer,
        device=device,
        test_dataloader=dataloaders['test'],
    )
    results_to_save = {'accuracy': test_results['accuracy']}
    save_test_results_to_json(
        results_to_save, 
        f"{LOGGING_FILEPATHS['TEST_RESULTS_DIRECTORY_PATH']}/test_results.json", 
        info=True
    )
    log.info('Testing ended!')

    log.info('Showing example predictions for Test Data...')
    for i in range(20):
        log.info(f'Example {i}:')
        log.info(f'Input text:\n\t {test_results["input_text"][i]}')
        log.info(f'Target text:\n\t {test_results["target_text"][i]}')
        log.info(f'Predicted text:\n\t {test_results["predicted_text"][i]}')
        log.info('######')

    log.info('Finished Pre-Training!')

    pass


if __name__ == "__main__":
    main()
