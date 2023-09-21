###################### Imports #################################

import logging
import torch
import os


###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


###################### Functions #################################

def cache_dict_as_pt(d: dict, filepath: str):
    """
    Save the dictionary as a .pt file.

    Parameters
    ----------
    d : dict
        The dictionary to save.
    filepath : str
        The path where the dictionary should be saved.

    Raises
    ------
    ValueError
        If d is not a dictionary or filepath is not a string.
    """
    if not isinstance(d, dict):
        raise ValueError('The input d must be a dictionary')
    if not isinstance(filepath, str):
        raise ValueError('The filepath must be a string')
    torch.save(d, filepath)

def load_dict_from_pt(filepath: str) -> dict:
    """
    Load the .pt file as a dictionary.

    Parameters
    ----------
    filepath : str
        The path of the .pt file.

    Returns
    -------
    dict
        The loaded dictionary.

    Raises
    ------
    ValueError
        If filepath is not a string or the loaded object is not a dictionary.
    FileNotFoundError
        If the file does not exist.
    """
    if not isinstance(filepath, str):
        raise ValueError('The filepath must be a string')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'The file {filepath} does not exist')

    d = torch.load(filepath)

    if not isinstance(d, dict):
        raise ValueError('The loaded object is not a dictionary')

    return d
