###################### Imports #################################

import logging
import os
import gc
import torch
from typing import Callable, Dict, List
import json
import copy


###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


###################### Functions #################################

def save_test_results_to_json(results_to_save: Dict, filepath: str, info: bool = False) -> None:
    """
    Saves a dictionary to a JSON file.

    Parameters
    ----------
    results_to_save : Dict
        The dictionary to save to the filepath.
    filepath : str
        Path of the file to save the results to.
    info : (bool, optional)
        Controls whether to log information about the filepath. (default: False)
    """
    try:
        # Create the directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Save the dictionary to a JSON file
        with open(filepath, 'w') as fp:
            json.dump(results_to_save, fp)
        
        if info:
            log.info(f"Test results saved to {filepath}")
    except Exception as e:
        log.info(f"Error saving test results: {e}")

def convert_function_values_to_name(config: dict) -> dict:
    """
    Converts the 'function' values in a nested dictionary to their respective names.
    """
    new_config = copy.deepcopy(config)

    for key, value in new_config.items():
        if isinstance(value, dict) and 'function' in value:
            value['function'] = value['function'].__name__

    return new_config

def save_configurations(dicts: List[dict], filepath: str, info: bool = False) -> None:
    """
    Saves a list of dictionaries to a JSON file.

    Parameters
    ----------
    dicts : List[dict]
        The dictionaries to save to the filepath.
    filepath : str
        Path of the file to save configurations to.
    info : (bool, optional)
        Controls whether to log information about the filepath. (default: False)
    """
    try:
        # Create the directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Save the dictionaries to the file
        with open(filepath, 'w') as fp:
            json.dump(dicts, fp)
        
        if info:
            log.info(f"Configurations saved to {filepath}")
    except Exception as e:
        log.info(f"Error saving configurations: {e}")
        

def load_configurations(filepath: str):
    """
    Load a dictionary from a JSON file.

    Parameters
    ----------
    filepath : str 
        Path of the file to load from.

    Returns
    -------
    The loaded dictionary, or list of dictionaries.
    """
    try:
        with open(filepath, 'r') as fp:
            dicts = json.load(fp)
        return dicts
    except Exception as e:
        log.info(f"Error loading configurations: {e}")



def log_constants(constants: dict) -> None:
    """
    Logs the defined constants and their values.

    Parameters
    ----------
    constants : dict
        A dictionary containing constant names as keys and their respective values.

    Returns
    -------
    None

    Examples
    --------
    >>> constants = {'PI': 3.14159, 'GRAVITY': 9.8}
    >>> log_constants(constants)
    Defined PI = 3.14159
    Defined GRAVITY = 9.8
    """
    for constant_name, constant_value in constants.items():
        log.info("Defined %s = %s", constant_name, constant_value)

    return

def report_gpu() -> None:
    """
    Reports GPU processes and clears GPU memory cache.

    Returns
    -------
    None

    Notes
    -----
    This function logs the GPU processes using `torch.cuda.list_gpu_processes()`
    and then performs garbage collection using `gc.collect()` to release any
    unreferenced GPU memory. Finally, it clears the GPU memory cache using
    `torch.cuda.empty_cache()`.

    Examples
    --------
    >>> report_gpu()
    GPU processes: [process1, process2, ...]
    """

    gpu_processes = torch.cuda.list_gpu_processes()
    log.info("GPU processes: %s", gpu_processes)
    gc.collect()
    torch.cuda.empty_cache()
    return

def get_from_map(item_type, item_map: Dict[str, Callable], **kwargs):
    """
    Helper method for getting an item from a map based on a type.
    """
    if item_type in item_map:
        return item_map[item_type](**kwargs)
    else:
        raise ValueError(f"Invalid type: {item_type}. Supported types are: {', '.join([item.name for item in item_map.keys()])}")
