###################### Imports #################################

import logging
import os
import pandas as pd
from typing import Dict
import pyarrow.parquet as pq


###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


###################### Functions #################################

def get_bloomberg_data(data_filepath: str, rows: int = None) -> pd.DataFrame:
    """
    TBC
    """
    parquet_file = pq.ParquetFile(data_filepath)

    chunks = []

    row_count = 0

    for i, batch in enumerate(parquet_file.iter_batches()):
        log.info(f"Retrieving batch {i}")
        batch_df = batch.to_pandas()
        chunks.append(batch_df)

        if rows:
            row_count += len(batch_df)
            if row_count > rows:
                break

    df = pd.concat(chunks, ignore_index=True)

    if rows is not None:
        return df.head(rows)
    else:
        return df

def get_reuters_data(data_folder: str, rows: int = None) -> pd.DataFrame:
    """
    Retrieves data from tab-separated files stored in the specified data_folder.

    Parameters
    ----------
    data_folder : str
        Path to the folder containing the tab-separated data files.

    rows : int, optional
        Number of rows to retrieve from the data files. If not specified, 
        all rows will be retrieved. (default: None)

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the data from the tab-separated files.

    Notes
    -----
    The files in the `data_folder` are assumed to be tab-separated with a header row.

    If the `rows` parameter is specified, the function retrieves only the first `rows` 
    number of rows from the data. The retrieval stops if the total number of rows across 
    all files exceeds `rows`.

    If `rows` is not specified, all rows from the data files are retrieved.

    Examples
    --------
    >>> data_folder = 'path/to/data_folder'
    >>> df = get_reuters_data(data_folder, rows=5)
    >>> print(df)
       Column1  Column2  Column3
    0        1        2        3
    1        4        5        6
    2        7        8        9
    3       10       11       12
    4       13       14       15
    """
    total_dfs = []
    row_count = 0
    for file_path in os.listdir(data_folder):
        df = pd.read_csv(os.path.join(data_folder, file_path), sep='\t', header=0)
        total_dfs.append(df)

        if rows:
            row_count += len(df)
            if row_count > rows:
                break
        
    reuters_df = pd.concat(total_dfs)

    if rows is not None:
        return reuters_df.head(rows)
    else: 
        return reuters_df

def get_data(data_source: Dict[str, Dict[str, any]], verbose: bool = False) -> pd.DataFrame:
    """
    TBC
    """
    data_dfs = []

    get_data_functions = {
        'reuters': get_reuters_data,
        'bloomberg': get_bloomberg_data,
    }

    source = data_source['source']
    path = data_source['path']
    rows = data_source['rows']

    if source in get_data_functions:
        data = get_data_functions[source](path, rows)
        data_dfs.append(data)
    else:
        raise ValueError('Invalid Data Source Specified!')
    
    all_data_df = pd.concat(data_dfs)
    if verbose: 
        log.info(f'Number of rows retrieved: {len(all_data_df)}')
        
    return all_data_df
