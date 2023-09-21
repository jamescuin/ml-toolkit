from datasets import DatasetDict, Dataset
import logging
import re

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)

def remove_missing_values_from_dataset_dict(dataset_dict: DatasetDict, verbose: bool = True, target_col: str = 'title') -> DatasetDict:
    """
    Remove rows with missing titles from each Dataset in a DatasetDict.

    Parameters
    ----------
    dataset_dict : DatasetDict
        The DatasetDict containing Datasets to process. Each Dataset should contain the target column.
    verbose : bool, optional
        Whether to log the number of rows before and after the operation for each Dataset.
        (default: True)
    target_col : str, optional
        The target column to check for missing values.
        (default: 'title')

    Returns
    -------
    DatasetDict
        The processed DatasetDict where, for each Dataset, rows with missing target column values are removed.

    Raises
    ------
    ValueError
        If the target column is not present in any of the Datasets.
    """

    def remove_missing_values_from_dataset(dataset: Dataset) -> Dataset:
        if target_col not in dataset.column_names:
            raise ValueError(f"{target_col} column not found in dataset.")

        filtered_dataset = dataset.filter(lambda example: example[target_col] is not None and example[target_col] != "")
        
        if verbose:
            logging.info(f'Original Number of Rows in {name}: {len(dataset)}')
            logging.info(f'Number of Rows in {name} after Missing {target_col} values removed: {len(filtered_dataset)}')
        
        return filtered_dataset

    # Process each Dataset in the DatasetDict
    for name, dataset in dataset_dict.items():
        dataset_dict[name] = remove_missing_values_from_dataset(dataset)

    return dataset_dict

def remove_extra_whitespace(string: str) -> str:
    """
    Remove extra whitespace from a string.

    This function removes any leading or trailing whitespace from a string,
    and replaces any consecutive spaces within the string with a single space.

    Parameters
    ----------
    string : str
        The string to process.

    Returns
    -------
    str
        The processed string where extra whitespace has been removed.
    """
    string = string.strip()
    pattern = ' +'
    string = re.sub(pattern, ' ', string)
    return string

def remove_extra_whitespace_from_dataset_dict(dataset_dict: DatasetDict, target_col: str = 'title') -> DatasetDict:
    """
    """

    def apply_function_to_column(example):
        example[target_col] = remove_extra_whitespace(example[target_col])
        return example

    processed_dataset_dict = DatasetDict()
    for key, dataset in dataset_dict.items():
        if target_col not in dataset.column_names:
            raise ValueError(f"'{target_col}' column not found in {key} Dataset.")
        processed_dataset_dict[key] = dataset.map(apply_function_to_column)
    return processed_dataset_dict


def remove_duplicates_from_dataset_dict(dataset_dict: DatasetDict, same_dt: bool = True, verbose: bool = True, target_col: str = 'title', dt_col: str = 'ts') -> DatasetDict:
    """
    """
    
    def filter_duplicates(dataset: Dataset):
        
        seen = set()
        
        def is_duplicate(example):
            key = example[target_col]
            if same_dt:
                key += example[dt_col][:8]
            if key in seen:
                return False
            seen.add(key)
            return True

        filtered_dataset = dataset.filter(is_duplicate)
        
        if verbose:
            log.info(f'Original Number of Rows in {dataset.split}: {len(dataset)}')
            log.info(f'Number of Rows after Exact Duplicates removed in {dataset.split}: {len(filtered_dataset)}')

        return filtered_dataset

    processed_dataset_dict = DatasetDict()

    for key, dataset in dataset_dict.items():
        # Move the column check here, outside of the filter_duplicates function
        if target_col not in dataset.column_names:
            raise ValueError(f"{target_col} column not found in {key} Dataset.")
        if same_dt:
            if dt_col not in dataset.column_names:
                raise ValueError(f"{dt_col} column not found in {key} Dataset.")
        
        processed_dataset_dict[key] = filter_duplicates(dataset)

    return processed_dataset_dict

def only_keep_characters(string: str, mode: str = 'alphabet_numbers_punctuation') -> str:
    """
    Filter characters from a string based on a specified mode.

    This function removes any characters from a string that do not fit within 
    the specified mode's parameters. For instance, if the mode is 'alphabet_numbers_punctuation',
    only alphabetic characters, numbers, and certain punctuation marks will be retained.

    Parameters
    ----------
    string : str
        The string to filter.
    mode : str, optional
        The filtering mode. Options are 'alphabet_numbers_punctuation' or 'alphabet_punctuation'. 
        (default: 'alphabet_numbers_punctuation')

    Returns
    -------
    str
        The filtered string.

    Raises
    ------
    ValueError
        If the mode is not recognized.
    """
    patterns = {
        'alphabet_numbers_punctuation': r'[^A-Za-z\d+.,!?\'"\- \/%@]',
        'alphabet_punctuation': r'[^A-Za-z.,!?\'"\- \/%@]'
    }
    
    if mode not in patterns:
        raise ValueError(f"Mode {mode} not recognized!")

    pattern = patterns[mode]
    filtered_string = re.sub(pattern, '', string)
    return filtered_string

def only_keep_characters_in_dataset_dict(dataset_dict: DatasetDict, mode: str = 'alphabet_numbers_punctuation', target_col: str = 'title') -> DatasetDict:
    """
    Apply the `only_keep_characters` function to the 'title' column of a DatasetDict.

    (Rest of the docstring adapted for DatasetDict)
    """

    def apply_function_to_column(example):
        example[target_col] = only_keep_characters(example[target_col], mode=mode)
        return example

    processed_dataset_dict = DatasetDict()
    for key, dataset in dataset_dict.items():
        if target_col not in dataset.column_names:
            raise ValueError(f"{target_col} column not found in {key} Dataset.")
        processed_dataset_dict[key] = dataset.map(apply_function_to_column)
    return processed_dataset_dict