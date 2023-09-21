###################### Imports #################################
import logging
import os
import torch
import glob


###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)

###################### Functions #################################

def checkpoint_model(
        cycle: int, 
        epoch: int, 
        epochs_per_cycle: int,
        model, 
        optimizer,
        lr_scheduler,
        best_val_metric, 
        train_tracker, 
        val_tracker,
        checkpoint_directory_path: str,
        checkpoint_save_mode: str = None,
        ):
    """
    Saves model at specific filepath, depending on the save_mode specified.

    save_mode = 'distinct':
        -> Seperate file for each epoch. 
        The naming convention is f'checkpoint_epoch{epoch}.pt'

    save_mode = 'overwrite':
        -> Same file for each epoch, which is overwritten each time.
        The naming convention is 'checkpoint_epoch_latest.pt'

    save_mode = None:
        -> No file saved for any epoch.  
    """
    if checkpoint_save_mode is None:
        return
    
    checkpoint = {
            'cycle': cycle,
            'epoch': epoch,
            'epochs_per_cycle': epochs_per_cycle,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_val_metric': best_val_metric,  # loss in pretraining, f1 in sentiment
            'train_tracker': train_tracker,
            'val_tracker': val_tracker,
        }
    if checkpoint_save_mode == 'distinct':
        checkpoint_filename = f'checkpoint_epoch{epoch}.pt'
    elif checkpoint_save_mode == 'overwrite':
        checkpoint_filename = 'checkpoint_epoch_latest.pt'
    else: 
        raise ValueError('Incorrect save_mode specified!')
    
    log.info('Saving Checkpoint...')
    checkpoint_filepath = os.path.join(checkpoint_directory_path, checkpoint_filename)
    torch.save(checkpoint, checkpoint_filepath)
    log.info(f'Checkpoint saved at: {checkpoint_filepath}')
    return

def load_latest_model_checkpoint(checkpoint_directory_path, model, optimizer, lr_scheduler) -> dict:
    """
    Loads the latest model checkpoint for the specified model_checkpoint_directory_path directory. 
    """

    log.info(f'Using Most Recent Checkpoint from: {checkpoint_directory_path}')
    checkpoint_files = glob.glob(os.path.join(checkpoint_directory_path, 'checkpoint_epoch*.pt'))
    latest_checkpoint_file = max(checkpoint_files, key=os.path.getctime)
    log.info(f'Latest Checkpoint is: {latest_checkpoint_file}')

    # Load the latest checkpoint
    checkpoint = torch.load(latest_checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    start_cycle = checkpoint['cycle']
    start_epoch = checkpoint['epoch']
    epochs_per_cycle = checkpoint['epochs_per_cycle']
    best_val_metric = checkpoint['best_val_metric']  # Update best val metric: loss in pretraining, f1 in sentiment
    train_tracker = checkpoint['train_tracker'] 
    val_tracker = checkpoint['val_tracker']

    start_epoch += 1
    if start_epoch % (epochs_per_cycle) == 0:
        log.info('Moving to next cycle')
        start_cycle += 1

    return {
        'model': model,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'start_cycle': start_cycle,
        'start_epoch': start_epoch,
        'best_val_metric': best_val_metric,
        'train_tracker': train_tracker,
        'val_tracker': val_tracker,
    } 