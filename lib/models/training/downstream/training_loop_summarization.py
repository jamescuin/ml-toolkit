###################### Imports #################################
import logging
import os
import torch
from tqdm.auto import tqdm
import numpy as np
from lib.models.handlers.model_loader import ModelLoader
from lib.models.handlers.adapter_handler import AdapterHandler
from lib.models.handlers.model_optimization import ModelOptimization
from rouge import Rouge
from lib.models.model_checkpointing import checkpoint_model, load_latest_model_checkpoint
from torch.utils.tensorboard import SummaryWriter
import random


###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)

###################### Functions #################################

def split_total(total, num_splits):
    """
    Used to calculate the number of batches in each epoch, for a cycle. 
    """
    # calculate the base value for each split
    split_value = total // num_splits
    
    # calculate the remaining value
    remaining = total % num_splits
    
    # create a list of split values
    splits = [split_value for _ in range(num_splits)]
    
    # distribute the remaining value among the first 'remaining' splits
    for i in range(remaining):
        splits[i] += 1
    
    return splits

def validate(val_dataloader, model, device, tokenizer, show_progress_bars: bool = False, max_target_length: int = None, show_n_predictions: int = None):
    """
    Validates the model on the given validation dataloader and calculates the mean validation loss.

    Important: You have to use the generate() method for evaluation
        because you want to be blind to the text that exists, if you do a normal forward pass the model sees 
        the text it's supposed to output
    """
    if show_progress_bars:
        val_progress_bar = tqdm(range(len(val_dataloader)))

    val_rouge1_f = []
    val_rouge1_p = []
    val_rouge1_r = []

    labels = []
    predictions = []

    rouge = Rouge()
        
    with torch.no_grad():
        log.info(f'Total val batches: {len(val_dataloader)}')
        for idx, batch in enumerate(val_dataloader):
            log.info(f'Batch: {idx}')
            # Pass the inputs to GPU
            model_batch = {k: v.to(device) for k, v in batch.items() if k!= 'labels'}

            # Generate text given input
            outputs = model.generate(**model_batch, 
                                     max_length=max_target_length,
                                     repetition_penalty=2.5)
            
            batch['labels'] = torch.where(batch['labels'] == torch.tensor(-100), torch.tensor(tokenizer.pad_token_id), batch['labels'])
            
            # Decode the model predictions back to text
            pred_text = tokenizer.batch_decode(outputs, 
                                               skip_special_tokens=True)
            
            label_text = tokenizer.batch_decode(batch['labels'].detach().cpu().numpy(),
                                                skip_special_tokens=True)

            pred_text = [p if p not in ['.', ''] else '/' for p in pred_text]
            
            val_rouge_scores = rouge.get_scores(pred_text, label_text, avg=True)
            
            val_rouge1_f.append(val_rouge_scores['rouge-1']['f'])
            val_rouge1_p.append(val_rouge_scores['rouge-1']['p'])
            val_rouge1_r.append(val_rouge_scores['rouge-1']['r'])
            
            val_mean_rouge1_f = np.mean(val_rouge1_f)
            val_mean_rouge1_p = np.mean(val_rouge1_p)
            val_mean_rouge1_r = np.mean(val_rouge1_r)

            labels += label_text
            predictions += pred_text

            if show_progress_bars:
                val_progress_bar.update(1)
                val_progress_bar.set_postfix(rouge_1_f=val_mean_rouge1_f)
    
    if show_n_predictions > 0: 
        log.info(f'Printing {show_n_predictions} Val predictions...')

        # Create a list of indices for random sampling
        indices = list(range(len(labels)))
        # Randomly sample indices
        sampled_indices = random.sample(indices, show_n_predictions)

        for i in sampled_indices:
            
            log.info('Target')
            log.info(f'\t {labels[i]}')
            
            log.info('Prediction')
            log.info(f'\t {predictions[i]}')
            
            log.info('######')
    
    return val_mean_rouge1_f, val_mean_rouge1_p, val_mean_rouge1_r

def train_model_summarization(
        model,
        tokenizer,
        optimizer,
        lr_scheduler,
        train_dataloader,
        val_dataloader,
        device,
        num_cycles: int,
        checkpoint_directory_path: str,
        best_model_directory_path: str,
        tensorboard_directory_path: str,
        use_latest_checkpoint: bool = False,
        checkpoint_save_mode: str = 'overwrite',
        show_progress_bars: bool = False,
        epochs_per_cycle: int = 1,
        max_target_length: int = None,
        show_n_predictions: int = None,
):
    """
    TBC
    """
    if max_target_length is None: 
        max_target_length = tokenizer.model_max_length

    # Keep track of model performance
    train_tracker = {'rouge1_f': [], 'rouge1_p': [], 'rouge1_r': []}
    val_tracker = {'rouge1_f': [], 'rouge1_p': [], 'rouge1_r': []}

    best_val_rouge1_f = 0 # Keep track of the best score. This should increase as we train

    start_cycle = 0  # Updated if `use_latest_checkpoint` is True.
    epoch = 0  # Epoch number across all cycles.
    cycle_epoch = 0  # Epoch number for specific cycle.
    batches_processed = 0  # Count used to calculate epoch.
    start_cycle_batch_num = 0  # When loading a checkpointed model, references which batch number for the cycle we got up to. Otherwise we may overfit on data of first epochs.   
    epoch_sizes = split_total(len(train_dataloader), epochs_per_cycle)  # Number of batches per epoch.
    
    log.info(f'Batch sizes for Epochs: {epoch_sizes}')

    rouge = Rouge()

    # Create directory to save checkpoints of model, if it doesn't exist.
    if not os.path.exists(checkpoint_directory_path):
        log.info(f'Creating Checkpoint Directory at: {checkpoint_directory_path}')
        os.makedirs(checkpoint_directory_path)

    # If specified, use the latest checkpoint of the model
    if use_latest_checkpoint:
        try: 
            # Load the latest checkpoint
            checkpoint = load_latest_model_checkpoint(checkpoint_directory_path, model, optimizer)
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            epoch = checkpoint['start_epoch']
            start_cycle = checkpoint['start_cycle']
            best_val_rouge1_f = checkpoint['best_val_metric']
            train_tracker = checkpoint['train_tracker']
            val_tracker = checkpoint['val_tracker']
            cycle_epoch = epoch % epochs_per_cycle
            start_cycle_batch_num = sum(epoch_sizes[:cycle_epoch])  # The required number of batches to process for the current epoch. 

        except ValueError:
            log.info('No Checkpoint to Load!')
        
    # Create a TensorBoard summary writer
    writer = SummaryWriter(tensorboard_directory_path)

    log.info(f'Total number of Cycles to train over: {num_cycles}')
    log.info(f'Starting from Epoch {epoch} (Cycle {start_cycle} - Cycle Epoch {cycle_epoch})')

    for cycle in range(start_cycle, num_cycles):
        log.info(f'Currently on Cycle {cycle}')
        
        # ======================= Training Phase =======================
        if show_progress_bars:
            progress_bar = tqdm(range(len(train_dataloader)))
        
        # These are for tracking performance
        train_losses = []
        train_rouge1_f = []
        train_rouge1_p = []
        train_rouge1_r = []   

        labels = []
        predictions = [] 
        
        optimizer.zero_grad()
        
        for idx, model_batch in enumerate(train_dataloader):
            # If loading a checkpoint partway through a cycle, we resume training from where we got to. 
            if (idx < start_cycle_batch_num) & (cycle == start_cycle):
                continue

            # Pass the inputs to GPU
            model_batch = {k: v.to(device) for k, v in model_batch.items()}

            # Model forward pass
            outputs = model(**model_batch)
            
            # Backwards pass
            loss = outputs.loss
            loss.backward()
            
            # Keep track of loss        
            train_losses.append(loss.cpu().data.item())
            train_mean_l = np.mean(train_losses)

            # Set -100 to pad_token_id for decoding. 
            model_batch['labels'] = torch.where(model_batch['labels'] == torch.tensor(-100), torch.tensor(tokenizer.pad_token_id), model_batch['labels'])
            
            # Decode the model predictions back to text
            pred_text = tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(),
                                               skip_special_tokens=True)
            
            label_text = tokenizer.batch_decode(model_batch['labels'].detach().cpu().numpy(),
                                                skip_special_tokens=True)

            # Process predictions and labels 
            pred_text = [p if p not in ['.', ''] else '/' for p in pred_text]

            
            train_rouge_scores = rouge.get_scores(pred_text, label_text, avg=True)
            

            train_rouge1_f.append(train_rouge_scores['rouge-1']['f'])
            train_rouge1_p.append(train_rouge_scores['rouge-1']['p'])
            train_rouge1_r.append(train_rouge_scores['rouge-1']['r'])

            train_mean_rouge1_f = np.mean(train_rouge1_f)
            train_mean_rouge1_p = np.mean(train_rouge1_p)
            train_mean_rouge1_r = np.mean(train_rouge1_r)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            labels += label_text
            predictions += pred_text

            if show_progress_bars:
                progress_bar.update(1)
                progress_bar.set_postfix(loss=train_mean_l, rouge_1_f=train_mean_rouge1_f)
            
            batches_processed += 1

            # Check if epoch completed.
            if batches_processed == epoch_sizes[cycle_epoch]:
                log.info(f'\nEpoch {epoch} Completed! (Cycle {cycle} - Cycle Epoch {cycle_epoch}) ')

                if show_n_predictions > 0: 
                    log.info(f'Printing {show_n_predictions} Train predictions...')

                    # Create a list of indices for random sampling
                    indices = list(range(len(labels)))
                    # Randomly sample indices
                    sampled_indices = random.sample(indices, show_n_predictions)

                    for i in sampled_indices:
                        
                        log.info('Target')
                        log.info(f'\t {labels[i]}')
                        
                        log.info('Prediction')
                        log.info(f'\t {predictions[i]}')
                        
                        log.info('######')

                batches_processed = 0  # reset batch count for the next epoch
                train_losses = []  # reset stats for next epoch
                train_rouge1_f = []
                train_rouge1_p = []
                train_rouge1_r = []

                labels = []
                predictions = []
        
                train_tracker['rouge1_f'].append(train_mean_rouge1_f)
                train_tracker['rouge1_p'].append(train_mean_rouge1_p)
                train_tracker['rouge1_r'].append(train_mean_rouge1_r)
        
                writer.add_scalar('ROUGE-1 F/train', train_mean_rouge1_f, epoch)
                writer.add_scalar('ROUGE-1 Precision/train', train_mean_rouge1_p, epoch)
                writer.add_scalar('ROUGE-1 Recall/train', train_mean_rouge1_r, epoch)
                writer.add_scalar('Loss/train', train_mean_l, epoch)

                log.info(f'Epoch {epoch} Training ROUGE-1 F1: {train_mean_rouge1_f}, Training ROUGE-1 Precision: {train_mean_rouge1_p}, Training ROUGE-1 Recall: {train_mean_rouge1_r}')

                # ======================= Validation Phase =======================
                log.info('Validating...')
                val_mean_rouge1_f, val_mean_rouge1_p, val_mean_rouge1_r = validate(val_dataloader, model, device, tokenizer, show_progress_bars, max_target_length, show_n_predictions)
                log.info('Validation Phase Completed!')

                val_tracker['rouge1_f'].append(val_mean_rouge1_f)
                val_tracker['rouge1_p'].append(val_mean_rouge1_p)
                val_tracker['rouge1_r'].append(val_mean_rouge1_r)

                log.info(f'\nEpoch {epoch} Val ROUGE-1 F: {val_mean_rouge1_f}, Val ROUGE-1 Precision: {val_mean_rouge1_p}, Val ROUGE-1 Recall: {val_mean_rouge1_r}')

                writer.add_scalar('ROUGE-1 F/val', val_mean_rouge1_f, epoch)
                writer.add_scalar('ROUGE-1 Precision/val', val_mean_rouge1_p, epoch)
                writer.add_scalar('ROUGE-1 Recall/val', val_mean_rouge1_r, epoch)

                # If the validation F1 is better, save the model to disk
                if val_mean_rouge1_f > best_val_rouge1_f:
                    best_val_rouge1_f = val_mean_rouge1_f
                    log.info('Saving Improved Model...')
                    model.save_pretrained(best_model_directory_path)  
                    # Note: Do Not merge Adapter into model here as sets [param.requires_grad for param in model.parameters()] to all False.
                    log.info(f'Improved Model saved at: {best_model_directory_path}')
                    log.info(f'Saving Corresponding Tokenzier...')
                    tokenizer.save_pretrained(best_model_directory_path)
                    log.info(f'Tokenizer saved at: {best_model_directory_path}')

                #  Save current configuration of model.
                checkpoint_model(
                    cycle=cycle, 
                    epoch=epoch,
                    epochs_per_cycle=epochs_per_cycle,
                    model=model, 
                    optimizer=optimizer, 
                    best_val_metric=best_val_rouge1_f, 
                    train_tracker=train_tracker, 
                    val_tracker=val_tracker,
                    checkpoint_directory_path=checkpoint_directory_path,
                    checkpoint_save_mode=checkpoint_save_mode
                    )
                
                epoch += 1  # increment epoch
                cycle_epoch += 1  # increment cycle epoch
                if cycle_epoch == len(epoch_sizes):  # if we've done all splits, reset for the next cycle
                    cycle_epoch = 0
                
                log.info(f'Moving onto Epoch {epoch}...')
    
    # Close the TensorBoard writer.
    writer.close()

    return {
        'train_tracker': train_tracker, 
        'val_tracker': val_tracker,
        }

def train_model_summarization_kfold(
        dataloaders_kfold,
        pretrained_model_directory_path: str,
        checkpoint_directory_path: str,
        best_model_directory_path: str,
        pretrained_model_type: str,
        device,
        adapter_type: str,
        add_adapter: bool,
        merge_adapter: bool,
        optimizer_type,
        scheduler_type,
        learning_rate: bool = 3e-4,
        weight_decay: bool = 1e-4,
        num_warmup_steps: int = 0, 
        num_cycles: int = 10,
        use_latest_checkpoint: bool = True,
        checkpoint_save_mode: str = 'overwrite',
        show_progress_bars: bool = False,
        epochs_per_cycle: int = 1,
        max_target_length: int = None,
        model_alias: str = 'to_delete',
        show_n_predictions: int = None,
        train_n_folds = 'all',
):
    """
    TBC
    """
    training_results = {}

    # If train_n_folds is not 'all', limit the number of folds to be trained
    if train_n_folds != 'all':
        # Convert train_n_folds to an integer, considering the case when it's not 'all'
        num_folds_to_train = min(int(train_n_folds), len(dataloaders_kfold))
    else:
        num_folds_to_train = len(dataloaders_kfold)

    # for fold, fold_dataloaders in dataloaders_kfold.items():
    for fold, fold_dataloaders in list(dataloaders_kfold.items())[:num_folds_to_train]:

        # Model & Tokenizer Loading
        model_loader = ModelLoader(
            model_path=pretrained_model_directory_path,
            model_type=pretrained_model_type,
            device=device,
            )
        tokenizer = model_loader.load_tokenizer()
        fold_model = model_loader.load_model()

        # Adding Adapter (if specified)
        if add_adapter:
            adapter_handler = AdapterHandler(
                model=fold_model, 
                adapter_type=adapter_type,
                device=device
                )
            fold_model = adapter_handler.add_adapter_to_model()
    
        # Model Optimization
        num_training_steps = num_cycles * len(fold_dataloaders['train'])  # Total number of batches we train on. 
        
        model_optimization = ModelOptimization(
            model=fold_model,
            optimizer_type=optimizer_type, 
            scheduler_type=scheduler_type
            )
        optimizer = model_optimization.get_optimizer(
            learning_rate=learning_rate, 
            weight_decay=weight_decay
            )
        lr_scheduler = model_optimization.get_scheduler(
            optimizer=optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps)
        
        log.info(f'Training Loop for {fold}')
        fold_training_results = train_model_summarization(
            model=fold_model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=fold_dataloaders['train'],
            val_dataloader=fold_dataloaders['val'],
            device=device,
            num_cycles=num_cycles,
            checkpoint_directory_path=checkpoint_directory_path + f'/{fold}', 
            best_model_directory_path=best_model_directory_path + f'/{fold}',
            tensorboard_directory_path=f'/home/james/tensorboard/summarization/logdir/{model_alias}/{fold}',
            use_latest_checkpoint=use_latest_checkpoint,
            checkpoint_save_mode=checkpoint_save_mode,
            show_progress_bars=show_progress_bars,
            epochs_per_cycle=epochs_per_cycle,
            max_target_length=max_target_length,
            show_n_predictions=show_n_predictions,
        )
        training_results[f'{fold}'] = fold_training_results
    return training_results

