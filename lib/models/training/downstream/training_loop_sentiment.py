###################### Imports #################################
import logging
import os
import torch
from tqdm.auto import tqdm
import numpy as np
from lib.models.handlers.model_loader import ModelLoader
from lib.models.handlers.adapter_handler import AdapterHandler
from lib.models.handlers.model_optimization import ModelOptimization
from lib.models.model_checkpointing import checkpoint_model, load_latest_model_checkpoint
from torch.utils.tensorboard import SummaryWriter
import random
from sklearn.metrics import precision_recall_fscore_support

###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)

###################### Functions #################################

def get_last_learning_rate(lr_scheduler):
    last_lr = lr_scheduler.get_last_lr()[0]
    if torch.is_tensor(last_lr):
        last_lr = last_lr.item()
    return last_lr


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
    TEXT_TO_ID = {
        'negative': 0,
        'positive': 1,
        'neutral': 2,
        }
    
    if show_progress_bars:
        val_progress_bar = tqdm(range(len(val_dataloader)))

    # These are for tracking performance
    val_accuracies = []

    val_labels = []
    val_predictions = []
    val_predictions_text = []
        
    with torch.no_grad():
        for idx, batch in enumerate(val_dataloader):
            # Pass the inputs to GPU
            model_batch = {k: v.to(device) for k, v in batch.items() if k!= 'labels'}
            
            # Generate text given input
            outputs = model.generate(
                **model_batch, 
                max_length=max_target_length
            )
            
            batch['labels'] = torch.where(batch['labels'] == torch.tensor(-100), torch.tensor(tokenizer.pad_token_id), batch['labels'])

            # Decode the model predictions back to text
            pred_text = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
                )
            
            label_text = tokenizer.batch_decode(
                batch['labels'].detach().cpu().numpy(),
                skip_special_tokens=True
            )
            
            # Convert the text to numerical
            # We use the get() method because at the start T5 might produce text that is not quite right
            # and which might not map to any particular label, so we default to negative if doesn't predict
            y_true = np.array([TEXT_TO_ID.get(text, 0) for text in label_text])
            preds = np.array([TEXT_TO_ID.get(text, 0) for text in pred_text])

            val_acc = np.mean(y_true == preds)
            val_accuracies.append(val_acc)
            val_mean_acc = np.mean(val_accuracies)

            # Keep track of predictions for end of epoch calculations
            val_predictions.append(preds)
            val_labels.append(y_true)
            val_predictions_text.append(pred_text)

            if show_progress_bars:
                val_progress_bar.update(1)
                val_progress_bar.set_postfix(acc=val_mean_acc)
            
            val_labels = np.concatenate(val_labels, axis=0)
            val_predictions = np.concatenate(val_predictions, axis=0)
            val_predictions_text = np.concatenate(val_predictions_text, axis=0)
            val_prec, val_recall, val_f1, _ = precision_recall_fscore_support(
                val_labels,
                val_predictions,
                average='macro'
            )

            if show_n_predictions > 0: 
                log.info(f'Printing {show_n_predictions} Val predictions...')
                for i in range(show_n_predictions):
                    log.info('Target')
                    log.info(f'\t {val_labels[i]}')
                    
                    log.info('Prediction')
                    log.info(f'\t {val_predictions[i]}')

                    log.info('Prediction Text')
                    log.info(f'\t {val_predictions_text[i]}')
                    
                    log.info('######')

            return val_mean_acc, val_prec, val_recall, val_f1

def train_model_sentiment(
        model,
        tokenizer,
        optimizer,
        lr_scheduler,
        train_dataloader,
        val_dataloader,
        device,
        num_cycles: int,
        checkpoint_dir: str,
        best_model_dir: str,
        tensorboard_dir: str,
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
    
    # This dictionary provides a mapping from the text labels to integer labels for metric calculation
    TEXT_TO_ID = {
        'negative': 0,
        'positive': 1,
        'neutral': 2,
        }

    # Keep track of model performance
    train_tracker = {'acc': [], 'f1': [], 'prec': [], 'recall': []}
    val_tracker = {'acc': [], 'f1': [], 'prec': [], 'recall': []}

    best_val_f1 = 0 # Keep track of the best score. This should increase as we train

    start_cycle = 0  # Updated if `use_latest_checkpoint` is True.
    epoch = 0  # Epoch number across all cycles.
    cycle_epoch = 0  # Epoch number for specific cycle.
    batches_processed = 0  # Count used to calculate epoch.
    start_cycle_batch_num = 0  # When loading a checkpointed model, references which batch number for the cycle we got up to. Otherwise we may overfit on data of first epochs.   
    epoch_sizes = split_total(len(train_dataloader), epochs_per_cycle)  # Number of batches per epoch.
    
    log.info(f'Number of Batches in each Epoch: {epoch_sizes}')

    # Create directory to save checkpoints of model, if it doesn't exist.
    if not os.path.exists(checkpoint_dir):
        log.info(f'Creating Checkpoint Directory at: {checkpoint_dir}')
        os.makedirs(checkpoint_dir)

    # If specified, use the latest checkpoint of the model
    if use_latest_checkpoint:
        try: 
            # Load the latest checkpoint
            checkpoint = load_latest_model_checkpoint(checkpoint_dir, model, optimizer, lr_scheduler)
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            lr_scheduler = checkpoint['lr_scheduler']
            epoch = checkpoint['start_epoch']
            start_cycle = checkpoint['start_cycle']
            best_val_f1 = checkpoint['best_val_metric']
            train_tracker = checkpoint['train_tracker']
            val_tracker = checkpoint['val_tracker']
            cycle_epoch = epoch % epochs_per_cycle
            start_cycle_batch_num = sum(epoch_sizes[:cycle_epoch])  # The required number of batches to process for the current epoch. 

        except ValueError:
            log.info('No Checkpoint to Load!')
        
    # Create a TensorBoard summary writer
    writer = SummaryWriter(tensorboard_dir)

    log.info(f'Total number of Cycles to train over: {num_cycles}')
    log.info(f'Starting from Epoch {epoch} (Cycle {start_cycle} - Cycle Epoch {cycle_epoch})')

    for cycle in range(start_cycle, num_cycles):
        log.info(f'Currently on Cycle {cycle}')
        
        # ======================= Training Phase =======================
        if show_progress_bars:
            train_progress_bar = tqdm(range(len(train_dataloader)))
        
        # These are for tracking performance
        train_losses = []
        train_accuracies = []
        learning_rates = []

        train_labels = []
        train_predictions = []
        train_predictions_text = []
        
        optimizer.zero_grad()
        
        for idx, model_batch in enumerate(train_dataloader):
            # If loading a checkpoint partway through a cycle, we resume training from where we got to. 
            if (idx < start_cycle_batch_num) & (cycle == start_cycle):
                continue

            # Pass the inputs to GPU
            model_batch = {k: v.to(device) for k, v in model_batch.items()}
            # Replace padding token id's of the labels by -100 so it's ignored by the loss
            model_batch['labels'] = torch.where(model_batch['labels'] == tokenizer.pad_token_id, torch.tensor(-100), model_batch['labels'])

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

            # Convert the text to numerical
            # We use the get() method because at the start T5 might produce text that is not quite right
            # and which might not map to any particular label, so we default to negative if doesn't predict
            y_true = np.array([TEXT_TO_ID.get(text, 0) for text in label_text])
            preds = np.array([TEXT_TO_ID.get(text, 0) for text in pred_text])

            
            train_acc = np.mean(y_true == preds)
            train_accuracies.append(train_acc)
            train_mean_acc = np.mean(train_accuracies)

            # Keep track of predictions for end of epoch calculations
            train_predictions.append(preds)
            train_labels.append(y_true)
            train_predictions_text.append(pred_text)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            learning_rate = get_last_learning_rate(lr_scheduler)
            learning_rates.append(learning_rate)

            if show_progress_bars:
                train_progress_bar.update(1)
                train_progress_bar.set_postfix(loss=train_mean_l, acc=train_mean_acc)
            
            batches_processed += 1

            # Check if epoch completed.
            if batches_processed == epoch_sizes[cycle_epoch]:
                log.info(f'\nEpoch {epoch} Completed! (Cycle {cycle} - Cycle Epoch {cycle_epoch}) ')

                if show_n_predictions > 0: 
                    log.info(f'Printing {show_n_predictions} Train predictions...')

                    # Create a list of indices for random sampling
                    indices = list(range(len(train_labels)))
                    # Randomly sample indices
                    sampled_indices = random.sample(indices, show_n_predictions)

                    for i in sampled_indices:
                        
                        log.info('Target')
                        log.info(f'\t {train_labels[i]}')
                        
                        log.info('Prediction')
                        log.info(f'\t {train_predictions[i]}')

                        log.info('Prediction Text')
                        log.info(f'\t {train_predictions_text[i]}')
                        
                        log.info('######')
                
                train_tracker['acc'].append(train_mean_acc)
                train_labels = np.concatenate(train_labels, axis=0)
                train_predictions = np.concatenate(train_predictions, axis=0)

                train_prec, train_recall, train_f1, _ = precision_recall_fscore_support(
                    train_labels,
                    train_predictions,
                    average='macro',
                    zero_division=0
                )

                train_tracker['f1'].append(train_f1)
                train_tracker['prec'].append(train_prec)
                train_tracker['recall'].append(train_recall)

                log.info(f'Epoch {epoch} (Cycle {cycle} - Cycle Epoch {cycle_epoch}) -> Training F1: {train_f1}, Precision: {train_prec}, Recall: {train_recall}, Accuracy: {train_mean_acc}')

                writer.add_scalar('Accuracy/train', train_mean_acc, epoch)
                writer.add_scalar('F1/train', train_f1, epoch)
                writer.add_scalar('Precision/train', train_prec, epoch)
                writer.add_scalar('Recall/train', train_recall, epoch)


                batches_processed = 0  # reset batch count for the next epoch
                train_losses = []  # reset stats for next epoch
                train_accuracies = []

                train_labels = []
                train_predictions = []

                # ======================= Validation Phase =======================
                log.info('Validating...')
                val_mean_acc, val_prec, val_recall, val_f1 = validate(val_dataloader, model, device, tokenizer, show_progress_bars, max_target_length, show_n_predictions)
                log.info('Validation Phase Completed!')

                val_tracker['acc'].append(val_mean_acc)
                val_tracker['f1'].append(val_f1)
                val_tracker['prec'].append(val_prec)
                val_tracker['recall'].append(val_recall)

                log.info(f'Epoch {epoch} (Cycle {cycle} - Cycle Epoch {cycle_epoch}) -> Val F1: {val_f1}, Precision: {val_prec}, Recall: {val_recall}, Accuracy: {val_mean_acc}')

                writer.add_scalar('Accuracy/val', val_mean_acc, epoch)
                writer.add_scalar('F1/val', val_f1, epoch)
                writer.add_scalar('Precision/val', val_prec, epoch)
                writer.add_scalar('Recall/val', val_recall, epoch)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    log.info('Saving Improved Model...')
                    model.save_pretrained(best_model_dir)
                    tokenizer.save_pretrained(best_model_dir)
                    log.info(f'Improved Model saved at: {best_model_dir}')

                # Save current configuration of model.
                checkpoint_model(
                    cycle=cycle, 
                    epoch=epoch,
                    epochs_per_cycle=epochs_per_cycle,
                    model=model, 
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    best_val_metric=best_val_f1, 
                    train_tracker=train_tracker, 
                    val_tracker=val_tracker,
                    checkpoint_directory_path=checkpoint_dir,
                    checkpoint_save_mode=checkpoint_save_mode,
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

def train_model_sentiment_kfold(
        dataloaders_kfold,
        pretrained_model_directory_path: str,
        checkpoint_directory_path: str,
        best_model_directory_path: str,
        pretrained_model_type: str,
        tensorboard_dir: str,
        device,
        adapter_type: str,
        add_adapter: bool,
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

    log.info(f'Num folds to train: {num_folds_to_train}')

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
        fold_training_results = train_model_sentiment(
            model=fold_model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=fold_dataloaders['train'],
            val_dataloader=fold_dataloaders['val'],
            device=device,
            num_cycles=num_cycles,
            checkpoint_dir=checkpoint_directory_path + f'/{fold}', 
            best_model_dir=best_model_directory_path + f'/{fold}',
            tensorboard_dir=tensorboard_dir + f'/{fold}',
            use_latest_checkpoint=use_latest_checkpoint,
            checkpoint_save_mode=checkpoint_save_mode,
            show_progress_bars=show_progress_bars,
            epochs_per_cycle=epochs_per_cycle,
            max_target_length=max_target_length,
            show_n_predictions=show_n_predictions,
        )
        training_results[f'{fold}'] = fold_training_results
    return training_results

