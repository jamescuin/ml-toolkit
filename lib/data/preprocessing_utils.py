###################### Imports #################################

import logging
import pandas as pd
import numpy as np
import re
from typing import Dict


###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


###################### Functions #################################

def remove_rows_without_column_value(df: pd.DataFrame, verbose: bool = True, target_col: str = 'title') -> pd.DataFrame:
    """
    Remove rows with missing titles from a DataFrame.

    Parameters
    ----------
    reuters_df : pd.DataFrame
        The DataFrame to process. It should contain a 'title' column.
    verbose : bool, optional
        Whether to log the number of rows before and after the operation.
        (default: True)

    Returns
    -------
    pd.DataFrame
        The processed DataFrame where rows with missing 'title' are removed.

    Raises
    ------
    ValueError
        If the 'title' column is not present in the DataFrame.
    """
    if target_col not in df.columns:
        raise ValueError(f"{target_col} column not found in DataFrame.")

    filtered_df = df.dropna(subset=[target_col])

    if verbose:
        log.info(f'Original Number of Rows: {len(df)}')
        log.info(f'Number of Rows after Missing {target_col} values removed: {len(filtered_df)}')

    return filtered_df

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

def remove_extra_whitespace_from_column_values(df: pd.DataFrame, target_col: str = 'title') -> pd.DataFrame:
    """
    Apply the `remove_extra_whitespace` function to the 'title' column of a DataFrame.

    This function copies a DataFrame and applies the `remove_extra_whitespace` 
    function to its 'title' column.

    Parameters
    ----------
    reuters_df : pd.DataFrame
        The DataFrame to process. It should contain a 'title' column.

    Returns
    -------
    pd.DataFrame
        The processed DataFrame, where the 'title' column has been stripped of extra whitespace.

    Raises
    ------
    ValueError
        If the 'title' column is not present in the DataFrame.
    """
    if target_col not in df.columns:
        raise ValueError("'title' column not found in DataFrame.")

    processed_df = df.copy()
    processed_df[target_col] = processed_df[target_col].apply(lambda x: remove_extra_whitespace(x))
    return processed_df

def remove_exact_duplicate_column_values(df: pd.DataFrame, same_date: bool = True, verbose: bool = True, target_col: str = 'title', dt_col: str = 'ts') -> pd.DataFrame:
    """
    Remove exact duplicate titles from a DataFrame.

    This function removes duplicate titles from the DataFrame based on whether the 
    'title' and optionally 'date' fields are the same. The 'date' is considered 
    if `same_date` is True and is extracted from the 'ts' field in the format 'YYYYMMDD'.

    Parameters
    ----------
    reuters_df : pd.DataFrame
        The DataFrame to process. It should contain a 'title' and a 'ts' column.
    same_date : bool, optional
        Whether to consider the 'date' field for detecting duplicates.
        (default: True)
    verbose : bool, optional
        Whether to log the number of rows before and after the operation. 
        (default: True)

    Returns
    -------
    pd.DataFrame
        The processed DataFrame where duplicate rows based on 'title' 
        and optionally 'date' are removed.

    Raises
    ------
    ValueError
        If the 'title' or 'ts' columns are not present in the DataFrame.
    """
    if target_col not in df.columns or dt_col not in df.columns:
        raise ValueError(f"{target_col} and/or {dt_col} column not found in DataFrame.")
    
    duplicate_subset = [target_col]

    if same_date:
        df['date'] = df[dt_col].apply(lambda x: x[0:8])
        duplicate_subset.append('date')

    processed_df = df.drop_duplicates(subset=duplicate_subset)

    if verbose:
        log.info(f'Original Number of Rows: {len(df)}')
        log.info(f'Number of Rows after Exact Duplicates removed: {len(processed_df)}')

    return processed_df

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

def only_keep_characters_in_column_values(df: pd.DataFrame, mode: str = 'alphabet_numbers_punctuation', target_col: str = 'title') -> pd.DataFrame:
    """
    Apply the `only_keep_characters` function to the 'title' column of a DataFrame.

    This function copies a DataFrame and applies the `only_keep_characters` function 
    to its 'title' column based on a specified mode. 

    Parameters
    ----------
    reuters_df : pd.DataFrame
        The DataFrame to process. It should contain a 'title' column.
    mode : str, optional
        The filtering mode to use in the `only_keep_characters` function.
        Options can be found in `only_keep_characters` documentation.
        (default: 'alphabet_numbers_punctuation')

    Returns
    -------
    pd.DataFrame
        The processed DataFrame, where the 'title' column has been filtered based 
        on the specified mode.

    Raises
    ------
    ValueError
        If the 'title' column is not present in the DataFrame.
    """
    if target_col not in df.columns:
        raise ValueError(f"{target_col} column not found in DataFrame.")

    processed_df = df.copy()
    processed_df[target_col] = processed_df[target_col].apply(lambda x: only_keep_characters(x, mode=mode))
    return processed_df

def get_tokenized_title_lengths_nth_percentile(reuters_df: pd.DataFrame, tokenizer, n: int, verbose: bool = True) -> int:
    """
    Calculates the length of the title at the nth percentile in a DataFrame.

    Parameters
    ----------
    reuters_df : pd.DataFrame
        The DataFrame containing the titles.
    tokenizer
        TBC
    n : int
        The percentile value (in integer form) at which to calculate the title length.
    verbose : bool, optional
        If True, log information about the nth percentile length. (default: True)

    Returns
    -------
    int
        The tokenized title lengths at the specified percentile.

    """
    token_lengths = reuters_df['title'].apply(lambda title: len(tokenizer.encode(title)))
    quantile_lenth = np.quantile(token_lengths, n / 100)

    if verbose:
        log.info(f'{n}th percentile length: {np.quantile(token_lengths, n / 100)}')

    return quantile_lenth

def remove_titles_outside_token_length_percentiles(reuters_df: pd.DataFrame, tokenizer, n_lower: int = None, n_upper: int = None, verbose: bool = True) -> pd.DataFrame:
    """
    Removes titles from a DataFrame that fall outside the specified token length percentiles.

    Parameters
    ----------
    reuters_df : pd.DataFrame
        The DataFrame containing the titles to be processed.
    tokenizer
        TBC
    n_lower : int, optional
        The lower percentile value (in integer form) to remove titles below. (default: None)
    n_upper : int, optional
        The upper percentile value (in integer form) to remove titles above. (default: None)
    verbose : bool, optional
        If True, log information about the number of rows before and after outliers are removed. (default: True)

    Returns
    -------
    pd.DataFrame
        The processed DataFrame with titles filtered based on the specified percentiles, for tokenized title lenghts.
    """
    processed_df = reuters_df.copy()

    # Apply the tokenizer to each title and calculate token length
    processed_df['tokens_length'] = processed_df['title'].apply(lambda title: len(tokenizer.encode(title)))

    if n_lower is not None:
        length_lower = get_tokenized_title_lengths_nth_percentile(reuters_df, tokenizer, n_lower, verbose=verbose)  # Use reuters_df here as want percentiles of original df.
        processed_df = processed_df[processed_df['tokens_length'] > length_lower]

    if n_upper is not None:
        length_upper = get_tokenized_title_lengths_nth_percentile(reuters_df, tokenizer, n_upper, verbose=verbose)
        processed_df = processed_df[processed_df['tokens_length'] < length_upper]

    if verbose:
        log.info(f'Original Number of Rows: {len(reuters_df)}')
        log.info(f'Number of Rows after Titles Outside Percentiles removed: {len(processed_df)}')

    return processed_df

def apply_preprocessing_functions(df: pd.DataFrame, config: Dict[str, Dict[str, any]], verbose: bool = True) -> pd.DataFrame:
    """
    Applies a series of preprocessing functions specified in the `config` on `df`. 
    The function also logs information about the number of rows in the dataframe before 
    and after preprocessing if verbose is set to True.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe to be preprocessed.
    config : Dict[str, Dict[str, any]]
        A dictionary with function names as keys. The values should be another
        dictionary with keys 'function' and 'args'. 'function' should be a 
        callable that takes a dataframe and some arguments, and 'args' should
        be a dictionary of arguments to pass to the function.
    verbose : bool, optional
        A flag to enable logging of information about the number of rows 
        before and after preprocessing. (default: True)

    Returns
    -------
    pd.DataFrame
        The preprocessed dataframe.

    Notes
    -----
    This function does not modify the input dataframe, it operates on a 
    copy of the dataframe.
    """
    processed_df = df.copy()
    for func_name, func_config in config.items():
        func = func_config['function']
        args = func_config['args']
        processed_df = func(processed_df, **args)
    
    if verbose:
        log.info(f'Number of Rows before All Pre-Processing: {len(df)}')
        log.info(f'Number of Rows after All Pre-Processing: {len(processed_df)}')

    return processed_df

def remove_column_values_under_n_tokens(df: pd.DataFrame, tokenizer, max_tokens: int = 10, target_col: str = 'title') -> pd.DataFrame:
    """
    Returns the df, with rows corresponding to titles shorter than `max_tokens` tokens removed. 
    """

    df['tokens_length'] = df[target_col].apply(lambda x: len(tokenizer.encode(x)))
    df = df[df['tokens_length'] >= max_tokens]
    
    return df