###################### Imports #################################

import logging
import numpy as np
import pandas as pd
from typing import Tuple


###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


###################### Functions #################################

def get_trainable_parameters(model) -> str:
    """
    Gets the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"


def get_token_lengths_nth_percentile(df: pd.DataFrame, tokenizer, n: int = 99) -> int:
    """
    Returns the nth_percentile of token lenghts, given the df and tokenizer passed. 
    """
    
    # Go through the text and tokenize it, keeping track of the length of tokens
    token_lengths = []
    for row in df.itertuples():
        tokenized_inputs = tokenizer.batch_encode_plus(
                [row.Sentence],
                )
        token_lengths.append(len(tokenized_inputs['input_ids'][0]))
    return int(np.quantile(token_lengths, n / 100))
    

def compute_input_and_target_lengths(
    inputs_length: int,
    noise_density: float,
    mean_noise_span_length: float,
    extra_tokens_per_span_inputs: int = 1,
    extra_tokens_per_span_targets: int = 1,
    verbose: bool = False,
) -> Tuple[int, int]:
    """
    Computes the required token length for inputs and the required token length for targets, for masked language modelling.

    It assumes that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens, 
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.

    Parameters
    ----------
    inputs_length : int
        The desired length of the tokenized inputs sequence.
    noise_density : float
        The density of the noise in the tokens, as a float between 0 and 1.
    mean_noise_span_length : float
        The average length of a span of noise tokens.
    extra_tokens_per_span_inputs : int, optional
        The number of extra tokens for each span in the input. (default: 1)
    extra_tokens_per_span_targets : int, optional
        The number of extra tokens for each span in the target. (default: 1)
    verbose : bool, optional
        Whether to log the tokens_length, inputs_length, targets_length, 
        noise_density, and mean_noise_span_length. (default: False)

    Returns
    -------
    Tuple[int, int]
        The required lengths for input and target token sequences to match the desired inputs_length after masking is applied. 

    Raises
    ------
    ValueError
        If the noise_density is not between 0 and 1.

    Notes
    -----
    This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
    """
    if not 0 <= noise_density <= 1:
        raise ValueError("noise_density should be between 0 and 1.")

    def _tokens_length_to_inputs_length_targets_length(tokens_length: int) -> tuple:
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        return (
            num_nonnoise_tokens + num_noise_spans * extra_tokens_per_span_inputs + 1,
            num_noise_tokens + num_noise_spans * extra_tokens_per_span_targets + 1)

    tokens_length = inputs_length

    while (_tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length):
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1

    if verbose:
        log.info(
            'inputs_length=%s noise_density=%s mean_noise_span_length=%s ',
            inputs_length, noise_density, mean_noise_span_length)
        log.info('Required length for input token sequence (before masking): %s', tokens_length)
        log.info('Required length for target token sequence: %s', targets_length)
    return tokens_length, targets_length
