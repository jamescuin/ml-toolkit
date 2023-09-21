###################### Imports #################################

import logging
import os
import pandas as pd
from typing import Dict
import pyarrow.parquet as pq
from datasets import load_dataset


###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


###################### Classes #################################

class DataGetter:
    """
    Handles loading of data from various sources.
    """

    def __init__(self, source: str, path: str, rows: int = None, verbose: bool = True):
        """
        Constructor for DataGetter.

        Parameters
        ----------
        source : str
            Source of the data. Current supported sources are 'reuters' and 'bloomberg'.
        path : str
            Path to the data. Could be a file or directory, based on the data source.
        rows : int, optional
            Number of rows to retrieve from the data. If not specified, 
            all rows will be retrieved.
        """
        self.source = source
        self.path = path
        self.rows = rows
        self.verbose = verbose

    def get_data(self, **kwargs) -> pd.DataFrame:
        """
        Returns the data from the source, with a certain number of rows if specified.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the data from the source.

        Raises
        ------
        ValueError
            If an unsupported source is provided.
        """
        log.info(f'Getting data from: {self.path}...')
        data_getter_map = {
            'reuters': self.get_reuters_data,
            'bloomberg': self.get_bloomberg_data,
            'mc4_nl_cleaned': self.get_hf_data,
            'cnn_dailymail_dutch': self.get_hf_data,
        }

        if self.source in data_getter_map:
            data = data_getter_map[self.source](self.path, **kwargs)
        else:
            raise ValueError(f'Invalid Data Source Specified: {self.source}')

        if self.verbose: 
            log.info(f'Number of rows retrieved: {len(data)}')

        return data

    def get_bloomberg_data(self, data_filepath: str) -> pd.DataFrame:
        """
        Retrieves data from a Bloomberg Parquet file stored in the specified path.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the data from the Parquet file.
        """
        parquet_file = pq.ParquetFile(data_filepath)
        chunks = []
        row_count = 0

        for i, batch in enumerate(parquet_file.iter_batches()):
            log.info(f"Retrieving batch {i}")
            batch_df = batch.to_pandas()
            chunks.append(batch_df)

            if self.rows:
                row_count += len(batch_df)
                if row_count >= self.rows:
                    break

        df = pd.concat(chunks, ignore_index=True)

        return df.head(self.rows) if self.rows is not None else df

    def get_reuters_data(self, data_folder: str) -> pd.DataFrame:
        """
        Retrieves data from a Reuters tab-separated files stored in the specified path.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the data from the tab-separated files.
        """
        total_dfs = []
        row_count = 0

        for file_path in os.listdir(data_folder):
            df = pd.read_csv(os.path.join(data_folder, file_path), sep='\t', header=0)
            total_dfs.append(df)

            if self.rows:
                row_count += len(df)
                if row_count >= self.rows:
                    break

        reuters_df = pd.concat(total_dfs)

        return reuters_df.head(self.rows) if self.rows is not None else reuters_df
    
    def get_hf_data(
        self, 
        path: str, 
        cache_dir: str, 
        train_split: str = 'train', 
        val_split: str = 'validation', 
        test_split: str = 'test',
        config: str = None,
    ) -> pd.DataFrame:
        """
        This function loads datasets from the specified path and combines the train, validation, and test splits into a single pandas DataFrame.

        Parameters
        ----------
        path (str): The path to the dataset.
        cache_dir (str): The directory where the cached datasets should be stored.
        train_split (str, optional): The name of the training split. Defaults to 'train'.
        val_split (str, optional): The name of the validation split. Defaults to 'validation'.
        test_split (str, optional): The name of the test split. Defaults to 'test'.
        config (str, optional): The specific configuration of the dataset to load. Defaults to None.

        Raises
        ------
        Exception: An error occurred while loading the dataset.
        """
        splits = [train_split, val_split, test_split]
        all_data = []

        # Check types of inputs
        if not all(isinstance(split, (str, type(None))) for split in splits):
            raise TypeError("All splits should be of str type.")
        
        for split in splits:
            if split is not None:
                try:
                    if config is None:
                        dataset = load_dataset(path, split=split, cache_dir=cache_dir)
                    else:
                        dataset = load_dataset(path, config, split=split, cache_dir=cache_dir)
                    all_data.append(dataset.to_pandas())
                except Exception as e:
                    log.info(f"An error occurred while loading the {split} dataset: {e}")
                    continue

        df = pd.concat(all_data, ignore_index=True)

        return df.head(self.rows) if self.rows is not None else df
        
