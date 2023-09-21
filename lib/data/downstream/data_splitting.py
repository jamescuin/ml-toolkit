###################### Imports #################################
import logging
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Dict, List
from tqdm import tqdm


###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)

###################### Functions #################################

# We want kfold cross validation on sentiment task.
# Perform training/val loop n times, seperately for each fold.
# Then using the best model config from each fold loop, we test and get n test scores.
# Final test score is the average across these. Also calculate the std deviation. 



def get_datasplits_kfold(
        df: pd.DataFrame,
        target_col: str,
        num_folds: int = 5,
        stratify: bool = True,
        info: bool = True,
) -> List[Dict[str, pd.DataFrame]]:
    """
    Generates k-fold cross-validation data splits.

    This function splits a dataframe into training, validation, and testing sets for k-fold cross-validation.
    It supports both stratified and regular k-fold splitting.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe to split.
    target_col : str
        The name of the target column in the dataframe. This is used for stratification.
    num_folds : int, optional
        The number of folds for the cross-validation. (default: 5)
    stratify : bool, optional
        Whether to perform stratified k-fold splitting. If True, the distribution of the target variable is 
        preserved in each fold. If False, the data is split randomly. (default: True)
    info: bool
        Whether to print info about the splits. (default: True)

    Returns
    -------
    list
        A list of dictionaries, one for each fold. Each dictionary contains three keys: 'train', 'val', and 'test'. The value for each
        key is a dataframe corresponding to the training, validation, and testing set for that fold, respectively.
    """
    assert target_col in df.columns, f"{target_col} does not exist as a column in the DataFrame"
    
    fold_splits = []

    # If num_folds is 1, simply split data into train and test sets
    if num_folds == 1:
        if stratify:
            train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=0, stratify=df[target_col])
        else:
            train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=0)

        # Further split train_val_df into train and val sets
        if stratify:
            train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=0, stratify=train_val_df[target_col])
        else:
            train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=0)

        fold_splits.append({
            'train': train_df,
            'val': val_df,
            'test': test_df
        })

    # If num_folds is greater than 1, perform k-fold cross-validation
    else:
        if stratify:
            kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)
        else:
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)

        for train_val_indices, test_indices in kf.split(X=df, y=df[target_col] if stratify else None):
            train_val_df = df.iloc[train_val_indices]
            test_df = df.iloc[test_indices]

            if stratify:
                train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=0, stratify=train_val_df[target_col])
            else:
                train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=0)

            fold_splits.append({
                'train': train_df,
                'val' : val_df,
                'test': test_df,
                })
        
    if info: 
        for fold_num, fold_split in enumerate(fold_splits):
            log.info(f'Splits for fold_{fold_num}:')
            for k, v in fold_split.items():
                log.info(f'Split - {k}:')
                vc = v[target_col].value_counts().reset_index()
                vc['pct'] = vc['count'] / vc['count'].sum()
                log.info(vc)

    return fold_splits

def tokenize_datasplits_kfold(
        datasplits_kfold: List[dict], 
        tokenizer, 
        max_input_length: int, 
        max_target_length: int,
        input_col: str,
        target_col: str,
    ):
    """
    Function to tokenize the datasets.

    Parameters
    ----------
    data_splits : list
        A list of dictionaries. Each dictionary contains the train, validation and test dataframes for one fold.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer to use.
    max_input_length : int
        Maximum length for input sequences.
    max_target_length : int
        Maximum length for target sequences.

    Returns
    -------
    dict
        A dictionary where keys are the fold names and values are the tokenized datasets for that fold.
    """
    assert target_col in datasplits_kfold[0]['train'].columns, f"{target_col} does not exist as a column in the DataFrames"
    
    splits = ['train', 'val', 'test']
    fold_tokenized_datasplits = {}

    for fold, fold_datasplits in enumerate(datasplits_kfold):
        fold_tokenized_datasplits[f'fold_{fold}'] = {split: None for split in splits}
        log.info(f'Tokenizing Fold: {fold}')
        for split in splits:  # 'train', 'val', 'test'
            data = {
                'input_ids': [], 
                'attention_mask': [], 
                'labels': []
                }
            for idx, row in tqdm(list(enumerate(fold_datasplits[split].itertuples())), desc=f'Tokenizing rows for {split} split'):
                try:
                    # First tokenize the input sentences, truncating to max length defined
                    tokenized_inputs = tokenizer.batch_encode_plus(
                        ["multi-class classification: " + getattr(row, input_col)],
                        max_length=max_input_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors="pt"
                    )
                    # These are the tokens for the words
                    data['input_ids'].append(tokenized_inputs['input_ids'].squeeze())
                    # This is just to mask out padding
                    data['attention_mask'].append(tokenized_inputs['attention_mask'].squeeze())

                    # Now tokenize the target sentence, they are only 1 token
                    tokenized_targets = tokenizer.batch_encode_plus(
                            [getattr(row, target_col)],
                            max_length=max_target_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt"
                    )
                    lm_labels = tokenized_targets["input_ids"].squeeze()
                    lm_labels[lm_labels[:] == tokenizer.pad_token_id] = -100
                    data['labels'].append(lm_labels)
                except Exception as e:
                    log.error(f'Error tokenizing row {idx} for {split} split: {e}')

            fold_tokenized_datasplits[f'fold_{fold}'][split] = data

    return fold_tokenized_datasplits


def get_dataloaders_kfold(
        tokenized_datasplits_kfold: Dict[str, Dict[str, dict]], 
        batch_size: int, 
        splits: List[str] = ['train', 'val', 'test'], 
        shuffle: bool = True, 
        drop_last: bool = False,
        ) -> Dict[str, Dict[str, DataLoader]]:
    """
    TBC
    """
    
    # Error checking
    if not tokenized_datasplits_kfold:
        raise ValueError("fold_tokenized_datasplits must not be None")

    data_loaders = {}

    for fold, fold_key in enumerate(tokenized_datasplits_kfold):
        data_loaders[fold_key] = {split: None for split in splits}
        for split in tokenized_datasplits_kfold[fold_key]:
            data = tokenized_datasplits_kfold[fold_key][split]
            
            # Now that we have input and labels, create a Dataloader object to iterate through the data
            dataset = Dataset.from_dict(data)
            dataset.set_format("torch")
            data_loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, drop_last=drop_last)
            
            data_loaders[f'fold_{fold}'][split] = data_loader

            logging.info(f"Created DataLoader for {fold}, split {split}")
    
    return data_loaders

