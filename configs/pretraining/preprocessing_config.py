import lib.data.preprocessing_utils as ppu

PREPROCESSING_CONFIG = {
    'remove_rows_without_column_value': {'function': ppu.remove_rows_without_column_value, 'args': {'target_col': 'text'}},
    'remove_extra_whitespace_from_column_values': {'function': ppu.remove_extra_whitespace_from_column_values, 'args': {'target_col': 'text'}},
    'remove_exact_duplicate_column_values': {'function': ppu.remove_exact_duplicate_column_values, 'args': {'same_date': False, 'target_col': 'text', 'dt_col': 'datetime'}},
    'only_keep_characters_in_column_values': {'function': ppu.only_keep_characters_in_column_values, 'args': {'mode': 'alphabet_punctuation', 'target_col': 'text'}},
    # 'remove_titles_outside_percentiles': {'function': ppu.remove_titles_outside_token_length_percentiles, 'args': {'n_lower': 40}},
}