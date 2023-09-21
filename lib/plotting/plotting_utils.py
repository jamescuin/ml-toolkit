###################### Imports #################################

import logging
import os
import matplotlib.pyplot as plt


###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


###################### Functions #################################

def visualise_training_performance(
        train_tracker: list,
        val_tracker: list,
        plot_directory_path: str = '/home/james/plots/temp/',
        plot_filename: str = 'to_delete.png',
        metric: str = 'loss'
):
    """
    Generates and saves a plot of the training and validation metrics throughout the training process.
    
    The function creates a directory, if it does not exist, and generates a plot showing how the specified
    metric changed over the epochs for both the training and validation sets, and then saves this plot
    in the provided directory with the given filename.
    
    Parameters
    ----------
    train_tracker : list
        A list containing the training metrics for each epoch.
    val_tracker : list
        A list containing the validation metrics for each epoch.
    plot_directory_path : str, optional
        The path to the directory where the plot will be saved. If the directory does not exist,
        it will be created. Default is '/home/james/plots/temp/'.
    plot_filename : str, optional
        The name of the file that the plot will be saved as. (default: 'to_delete.png')
    metric : str, optional
        The name of the metric to be plotted. This metric should exist in both the train_tracker 
        and val_tracker lists. (default: 'loss')
        
    Returns
    -------
    None
    """

    # Create directory to save checkpoints of model, if it doesn't exist.
    if not os.path.exists(plot_directory_path):
        log.info(f'Creating Checkpoint Directory at: {plot_directory_path}')
        os.makedirs(plot_directory_path)

    # Visualise model performance as epochs change.
    epochs = [i for i in range(len(val_tracker[metric]))]

    save_to_filepath = os.path.join(plot_directory_path, plot_filename)
    fig = plt.figure()
    plt.plot(epochs, train_tracker[metric])
    plt.plot(epochs, val_tracker[metric])
    plt.savefig(save_to_filepath)
    log.info(f'Saved Training Perfromace Plot at: {save_to_filepath}')
    return