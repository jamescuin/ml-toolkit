###################### Imports #################################

import logging
from typing import Dict, List
import os
import torch
from tqdm.auto import tqdm
import numpy as np
import pickle
from datetime import datetime as dt
from lib.models.model_checkpointing import checkpoint_model, load_latest_model_checkpoint
from torch.utils.tensorboard import SummaryWriter
from torch import nn


###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


###################### Functions #################################

"""
cycle = One whole pass through the training dataset.
epoch = A pass through a subset of the training dataset that we update weights and validate on.

In the below, we can specify how many epochs we wish to have per cycle.
"""

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

def train_model(
        model,
        tokenizer,
        optimizer,
        lr_scheduler,
        train_dataloader,
        val_dataloader,
        device,
        num_cycles: int, 
        checkpoint_directory_path: str,
        pretrained_model_directory_path: str,
        tensorboard_directory_path: str,
        use_latest_checkpoint: bool = False,
        checkpoint_save_mode: str = 'overwrite',
        show_progress_bars: bool = False,
        epochs_per_cycle: int = 1,
) -> Dict[str, List[float]]:
    """
    Trains the provided model using the given parameters and data loaders.

    Parameters
    ----------
    model
        The PyTorch model to train.
    optimizer : torch.optim.Optimizer
        The optimizer for the model.
    lr_scheduler : torch.optim.lr_scheduler
        The learning rate scheduler.
    train_dataloader : torch.utils.data.DataLoader
        The data loader for the training data.
    val_dataloader : torch.utils.data.DataLoader
        The data loader for the validation data.
    device : torch.device
        The device (CPU/GPU) to train the model on.
    num_epochs : int
        The number of times we wish to pass through the entire train_dataloader (often is all training dataset).
    checkpoint_directory_path : str
        The path to the directory where model checkpoints will be saved.
    pretrained_model_directory_path : str
        The path to the directory where the best performing (on the val data) pre-trained model will be saved.
    use_latest_checkpoint : bool, optional
        Whether to use the latest model checkpoint.(default: False)
    checkpoint_save_mode : str, optional
        Specifies how to save the model checkpoints. (default: 'overwrite')
    show_progress_bars : bool, optional
        Whether to display a progress bar for each epoch. (default: False)
    epochs_per_cycle: int, optional
        The number of times we validate our model per pass through the train_dataloader.
        The epochs are split as evenly as possible, via the `split_total` function.

    Returns
    -------
    Dict[str, List[float]]
        A dictionary containing training and validation metrics.
    """
    # Keep track of loss for both training and validation epochs.
    train_tracker = {'loss': []}
    val_tracker = {'loss': []}
     
    best_validation_loss = 1e20 # Keep track of the best validation loss. This will decrease as we train.
    
    start_cycle = 0  # Updated if `use_latest_checkpoint` is True.
    epoch = 0  # Epoch number across all cycles.
    cycle_epoch = 0  # Epoch number for specific cycle.
    batches_processed = 0  # Count used to calculate epoch.
    start_cycle_batch_num = 0  # When loading a checkpointed model, references which batch number for the cycle we got up to. Otherwise we may overfit on data of first epochs.   
    epoch_sizes = split_total(len(train_dataloader), epochs_per_cycle)  # Number of batches per epoch.
    
    batch_num = 0

    log.info(f'Total Batches: {sum(epoch_sizes)}')
    log.info(f'Number of Batches in Epochs: {epoch_sizes}')

    # Create a directory to keep track of the metrics.
    if not os.path.exists('metrics/pretraining/'):
        log.info(f'Creating Metrics Directory at: metrics/pretraining/')
        os.makedirs('metrics/pretraining/')

    # Create directory to save checkpoints of model, if it doesn't exist.
    if not os.path.exists(checkpoint_directory_path):
        log.info(f'Creating Checkpoint Directory at: {checkpoint_directory_path}')
        os.makedirs(checkpoint_directory_path)

    # If specified, use the latest checkpoint of the model.
    if use_latest_checkpoint:
        try: 
            checkpoint = load_latest_model_checkpoint(checkpoint_directory_path, model, optimizer, lr_scheduler)
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            lr_scheduler = checkpoint['lr_scheduler']
            epoch = checkpoint['start_epoch']
            start_cycle = checkpoint['start_cycle']
            best_validation_loss = checkpoint['best_val_metric']
            train_tracker = checkpoint['train_tracker']
            val_tracker = checkpoint['val_tracker']
            cycle_epoch = epoch % epochs_per_cycle
            start_cycle_batch_num = sum(epoch_sizes[:cycle_epoch])  # The required number of batches to process for the current epoch.

        except ValueError:
            log.info('No Checkpoint to Load!')

    # Create a TensorBoard summary writer, used to visualise loss over epochs.
    writer = SummaryWriter(tensorboard_directory_path)

    log.info(f'Total number of Cycles to train over: {num_cycles}')
    log.info(f'Epochs per Cycle: {epochs_per_cycle}')
    log.info(f'Starting from Epoch {epoch} (Cycle {start_cycle} - Cycle Epoch {cycle_epoch})')

    for cycle in range(start_cycle, num_cycles):
        log.info(f'Currently on Cycle {cycle}')

        # Training Phase
        if show_progress_bars:
            train_progress_bar = tqdm(range(len(train_dataloader)))
    
        train_losses = []
        learning_rates = []
        
        optimizer.zero_grad()
        
        for idx, model_batch in enumerate(train_dataloader):
            batch_num += 1
            # If loading a checkpoint partway through a cycle, we resume training from where we got to. 
            if (idx < start_cycle_batch_num) & (cycle == start_cycle):
                continue
                    
            model_batch = {k: v.to(device) for k, v in model_batch.items()}
            # Replace padding token id's of the labels by -100 so it's ignored by the loss
            model_batch['labels'] = torch.where(model_batch['labels'] == tokenizer.pad_token_id, torch.tensor(-100), model_batch['labels'])

            # Predict the masked tokens
            outputs = model(**model_batch)
            
            loss = outputs.loss
            loss.backward()  # Backwards pass
            
            train_losses.append(loss.cpu().data.item())
            mean_l = np.mean(train_losses)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            learning_rate = get_last_learning_rate(lr_scheduler)
            learning_rates.append(learning_rate)

            if show_progress_bars:
                train_progress_bar.update(1)
                train_progress_bar.set_postfix(loss=mean_l)  # Updates loss for each model_batch in progress bar.

            batches_processed += 1

            # Check if epoch completed.
            if batches_processed == epoch_sizes[cycle_epoch]:
                log.info(f'\nEpoch {epoch} Completed! (Cycle {cycle} - Cycle Epoch {cycle_epoch})')

                batches_processed = 0  # reset batch count for the next epoch
                train_losses = []  # reset loss for next epoch
        
                train_tracker['loss'].append(mean_l)
                writer.add_scalar('Train/Loss (epoch)', mean_l, epoch)
                writer.add_scalar('Learning rate', learning_rate, epoch)
                log.info(f'Epoch {epoch} Train Loss: {mean_l}')

                mean_val_l = validate(val_dataloader, model, device, tokenizer, show_progress_bars)

                val_tracker['loss'].append(mean_val_l)
                writer.add_scalar('Val/Loss (epoch)', mean_val_l, epoch)
                log.info(f'Epoch {epoch} Val Loss: {mean_val_l}')
        
                with open('metrics/pretraining/train_metrics.pkl', 'wb') as outp:
                        pickle.dump(train_tracker, outp)

                with open('metrics/pretraining/val_metrics.pkl', 'wb') as outp:
                    pickle.dump(val_tracker, outp)
        
                if mean_val_l < best_validation_loss:
                    best_validation_loss = mean_val_l
                    log.info('Saving Improved Model...')
                    model.save_pretrained(pretrained_model_directory_path)
                    log.info(f'Improved Model saved at: {pretrained_model_directory_path}')

                # Save current configuration of model.
                checkpoint_model(
                    cycle=cycle, 
                    epoch=epoch,
                    epochs_per_cycle=epochs_per_cycle,
                    model=model, 
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    best_val_metric=best_validation_loss, 
                    train_tracker=train_tracker, 
                    val_tracker=val_tracker,
                    checkpoint_directory_path=checkpoint_directory_path,
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

def validate(val_dataloader, model, device, tokenizer, show_progress_bars: bool = False):
    """
    Validates the model on the given validation dataloader and calculates the mean validation loss.
    """
    if show_progress_bars:
        val_progress_bar = tqdm(range(len(val_dataloader)))

    val_losses = []
        
    with torch.no_grad():
        for idx, model_batch in enumerate(val_dataloader):
            model_batch = {k: v.to(device) for k, v in model_batch.items()}
            # Replace padding token id's of the labels by -100 so it's ignored by the loss
            model_batch['labels'] = torch.where(model_batch['labels'] == tokenizer.pad_token_id, torch.tensor(-100).to(device), model_batch['labels'])

            outputs = model(**model_batch)

            loss = outputs.loss

            val_losses.append(loss.cpu().data.numpy())
            mean_l = np.mean(val_losses)

            if show_progress_bars:
                val_progress_bar.update(1)
                val_progress_bar.set_postfix(loss=mean_l)
    
    return mean_l