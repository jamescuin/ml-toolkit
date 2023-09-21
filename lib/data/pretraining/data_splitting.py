###################### Imports #################################

import logging
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, List


###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


###################### Functions #################################

def tokenize_function_reuters(examples: dict, tokenizer, expanded_inputs_length: int) -> dict:
    """
    Tokenizes a batch of text examples up to the specified maximum length.

    Parameters
    ----------
    examples : dict
        A dictionary containing the text examples under the 'text' key.
    tokenizer
        The tokenizer to be used.
    expanded_inputs_length : int
        The maximum length for the tokenized sequence. Sequences longer than this
        will be truncated, and sequences shorter will be padded.

    Returns
    -------
    dict
        A dictionary containing the tokenized 'input_ids'. 

    Examples
    --------
    >>> from transformers import BertTokenizer
    >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    >>> examples = {"text": ["Hello, world!", "Machine learning is great!"]}
    >>> max_length = 16
    >>> result = tokenize_function(examples, tokenizer, expanded_inputs_length)
    """
    tokenizer_out = tokenizer(
        text=examples["text"],
        return_attention_mask=True,
        max_length=expanded_inputs_length,
        padding="max_length",
        truncation=True,
    )

    return {"input_ids": tokenizer_out["input_ids"]}

def get_datasplits(
        df: pd.DataFrame,
        train_split_size: float = 0.8,
        val_split_size: float = 0.1,
        test_split_size: float = 0.1,
) -> Dict[str, pd.DataFrame]:
    """
    Split the provided DataFrame into train, validation, and test sets.

    The function splits the data into train, validation, and test sets according to the 
    provided split sizes. It returns a dictionary containing the three DataFrames.

    Parameters
    ----------
    reuters_df : pd.DataFrame
        A pandas DataFrame containing the data to be split.
    train_split_size : float, optional
        The size of the train set as a proportion of the total dataset. (default: 0.8)
    val_split_size : float, optional
        The size of the validation set as a proportion of the total dataset. (default: 0.1)
    test_split_size : float, optional
        The size of the test set as a proportion of the total dataset. (default: 0.1)

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary containing 'train', 'val', and 'test' keys, each mapped to the respective DataFrame.

    Raises
    ------
    ValueError
        If the provided split sizes do not add up to 1.
    """
    if train_split_size + val_split_size + test_split_size != 1:
        raise ValueError("Split sizes should add up to 1.")

    train_val_split_test_size = 1 - train_split_size
    val_test_split_test_size = test_split_size / (val_split_size + test_split_size)

    train_df, val_df = train_test_split(
        df, 
        test_size=train_val_split_test_size,
        random_state=0
    )

    val_df, test_df = train_test_split(
        val_df, 
        test_size=val_test_split_test_size,
        random_state=0
    )
    
    return {
        'train': train_df,
        'val' : val_df,
        'test': test_df,
        }

def create_datasplits_dict_reuters(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
) -> Dict[str, Dict[str, List[str]]]:
    """
    Create a dictionary of data splits from the provided DataFrames.

    This function creates a dictionary of 'train', 'val', and 'test' keys, each mapped to 
    another dictionary with the key 'text'. The 'text' key is mapped to a list of titles 
    from the respective DataFrame.

    Parameters
    ----------
    train_df : pd.DataFrame
        A pandas DataFrame containing the train data.
    val_df : pd.DataFrame
        A pandas DataFrame containing the validation data.
    test_df : pd.DataFrame
        A pandas DataFrame containing the test data.

    Returns
    -------
    Dict[str, Dict[str, List[str]]]
        A dictionary with 'train', 'val', and 'test' keys, each mapped to another dictionary 
        with the key 'text'. The 'text' key is mapped to a list of titles from the respective DataFrame.

    Example
    -------
    >>> train_df = pd.DataFrame({"title": ["Train title 1", "Train title 2"]})
    >>> val_df = pd.DataFrame({"title": ["Validation title 1"]})
    >>> test_df = pd.DataFrame({"title": ["Test title 1", "Test title 2", "Test title 3"]})
    >>> create_datasplits_dict_reuters(train_df, val_df, test_df)
    {
        'train': {'text': ['Train title 1', 'Train title 2']}, 
        'val': {'text': ['Validation title 1']}, 
        'test': {'text': ['Test title 1', 'Test title 2', 'Test title 3']}
    }
    """
    data_splits = ['train', 'val', 'test']
    data_frames = [train_df, val_df, test_df]
    data_dicts = {split: {'text': []} for split in data_splits}

    for split, df in zip(data_splits, data_frames): 
        data_dicts[split]['text'] = df['title'].tolist()

    return data_dicts

def get_tokenized_datasplits_reuters(
        datasplits: dict,
        tokenizer,
        expanded_inputs_length: int,
) -> dict:
    """
    TBC
    """
    data_dicts = create_datasplits_dict_reuters(datasplits['train'], datasplits['val'], datasplits['test'])

    tokenized_datasplits = {split: Dataset.from_dict(data_dict).map(
        tokenize_function_reuters,
        batched=True,
        fn_kwargs={
            'tokenizer': tokenizer,
            'expanded_inputs_length': expanded_inputs_length,
            },
            remove_columns=['text'],
            )
            for split, data_dict in data_dicts.items()
            }
    
    return tokenized_datasplits

def get_dataloaders(
        tokenized_datasplits: dict,
        data_collator,
        batch_size: int = 32,
) -> Dict[str, DataLoader]:
    """
    TBC
    """
    dataloaders = {split: DataLoader(
        datasplit,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator
        )
        for split, datasplit in tokenized_datasplits.items()
        }
    
    return dataloaders


def tokenize_function_bloomberg(examples, tokenizer, expanded_inputs_length: int):
    """
    TBC
    """

    tokenizer_out = tokenizer(
        text=examples["text"],
        return_attention_mask=False,
    )

    input_ids = tokenizer_out["input_ids"]

    concatenated_ids = np.concatenate(input_ids)

    total_length = concatenated_ids.shape[0]
    total_length = (total_length // expanded_inputs_length) * expanded_inputs_length

    concatenated_ids = concatenated_ids[:total_length].reshape(-1, expanded_inputs_length)
    result = {"input_ids": concatenated_ids}

    return result

def create_datasplits_dict_bloomberg(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
) -> Dict[str, Dict[str, List[str]]]:
    """
    TBC
    """
    data_splits = ['train', 'val', 'test']
    data_frames = [train_df, val_df, test_df]
    data_dicts = {split: {'text': []} for split in data_splits}

    for split, df in zip(data_splits, data_frames): 
        data_dicts[split]['text'] = df['text'].tolist()

    return data_dicts

def get_tokenized_datasplits_bloomberg(
        datasplits: dict,
        tokenizer,
        expanded_inputs_length: int,
) -> dict:
    """
    TBC
    """
    data_dicts = create_datasplits_dict_bloomberg(datasplits['train'], datasplits['val'], datasplits['test'])

    tokenized_datasplits = {split: Dataset.from_dict(data_dict).map(
        tokenize_function_bloomberg,
        batched=True,
        fn_kwargs={
            'tokenizer': tokenizer,
            'expanded_inputs_length': expanded_inputs_length,
            },
            remove_columns=['text'],
            )
            for split, data_dict in data_dicts.items()
            }
    
    return tokenized_datasplits


def get_tokenized_datasplits(
        data_source: str,
        datasplits: dict,
        tokenizer,
        expanded_inputs_length: int) -> dict:
    """
    TBC
    """
    tokenized_datasplits_functions = {
        'reuters': get_tokenized_datasplits_reuters,
        'bloomberg': get_tokenized_datasplits_bloomberg,
        'mc4_nl_cleaned': get_tokenized_datasplits_bloomberg,
    }

    selected_function = tokenized_datasplits_functions.get(data_source)

    if selected_function:
        return selected_function(
            datasplits=datasplits, 
            tokenizer=tokenizer, 
            expanded_inputs_length=expanded_inputs_length
        )
    else:
        raise ValueError('Invalid data source specified!')



