###################### Imports #################################


import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from transformers import BatchEncoding
from dataclasses import dataclass
from transformers import AutoTokenizer
import torch
import random
from transformers.data.data_collator import (
    DataCollatorMixin,
    _torch_collate_batch,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from collections.abc import Mapping
from torch.nn import functional as F

###################### Variables #################################


logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


###################### Functions #################################
def random_spans_noise_mask(length, mean_noise_span_length, noise_density):
    """
    A copy from https://github.com/EleutherAI/oslo/blob/main/oslo/transformers/tasks/data_t5_pretraining.py#L230 (inception)
    This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
    Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
        length: an int32 scalar (length of the incoming token sequence)
        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number
    Returns:
        a boolean tensor with shape [length]
    """

    orig_length = length

    num_noise_tokens = int(np.round(length * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
            num_items: an integer scalar > 0
            num_segments: an integer scalar in [1, num_items]
        Returns:
            a Tensor with shape [num_segments] containing positive integers that add
            up to num_items
        """
        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        np.random.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]])
        segment_id = np.cumsum(first_in_segment)
        # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(
        num_nonnoise_tokens, num_noise_spans
    )

    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2],
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros((length,), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return is_noise[:orig_length]

def get_data_collator(
        collator_config, 
        tokenizer,
        noise_density: float,
        mean_noise_span_length: float,
        input_length: int,
        target_length: int,
        ):
    """
    Function to return specified DataCollator. 
    """
    if collator_config["ACTIVE_COLLATOR"] == 'DataCollatorForT5MLMReuters':
        log.info(f'Retrieving DataCollatorForT5MLMReuters...')
        return DataCollatorForT5MLMReuters(
            tokenizer=tokenizer,
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
            input_length=input_length-1,  # As removing eos token from input_ids at start of collator
            target_length=target_length-1,  # As removing eos token from input_ids at start of collator
        )
    elif collator_config["ACTIVE_COLLATOR"] == 'DataCollatorForT5MLMBloomberg':
        log.info(f'Retrieving DataCollatorForT5MLMBloomberg...')
        return DataCollatorForT5MLMBloomberg(
            tokenizer=tokenizer,
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
            input_length=input_length,
            target_length=target_length,
        )
    elif collator_config["ACTIVE_COLLATOR"] == 'DataCollatorForT5UL2Bloomberg':
        log.info(f'Retrieving DataCollatorForT5UL2Bloomberg...')
        config = collator_config["DataCollatorForT5UL2Bloomberg"]
        return DataCollatorForT5UL2Bloomberg(
            tokenizer=tokenizer,
            r_denoising=config["R_DENOISING"],
            r_probability=config["R_PROB"],
            r_denoising_config=config["R_CONFIG"],
            s_denoising=config["S_DENOISING"],
            s_probability=config["S_PROB"],
            x_denoising=config["X_DENOISING"],
            x_probability=config["X_PROB"],
            x_denoising_config=config["X_CONFIG"],
        )
    else:
        raise ValueError('Invalid data collator type specified!')


###################### Classes #################################


'''
Here we define our data collators -- a mechanism for how the data in a batch is processed.
This determines how a sequence of input_ids are masked, and is also responsible for creating the
corresponding labels.
'''


@dataclass
class DataCollatorForT5MLMReuters:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `input_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    
    Attributes
    ----------
    tokenizer : transformers.AutoTokenizer
        The tokenizer used for encoding the data.
    noise_density : float
        The probability with which to (randomly) mask tokens in the input.
    mean_noise_span_length : float
        The average span length of the masked tokens.
    input_length : int
        The expected input length after masking.
    target_length : int
        The expected target length after masking.
    """
    tokenizer: AutoTokenizer
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:

        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )

        # Get the input_ids, before masking, for each input sequence in the batch.
        input_ids = batch["input_ids"]

        # Get the batch size and the size of the input_ids, before masking, for each input seqence in the batch.
        batch_size, expandend_input_length = input_ids.shape

        # For each input sequence in the batch, get a boolean array indicating whether to mask or not. 
        # Masking for each input sequence is determined by the `random_spans_noise_mask` method. 
        mask_booleans = [self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)]

        # As random_spans_noise_mask always masks same token numbers, if counting from right. 
        # mask_booleans = shift_list_elements(mask_booleans, count_consecutive_groups(mask_booleans))

        inputs_mask_indices = np.asarray(mask_booleans)

        # For each input sequence, get a boolean array, opposite to the `inputs_mask_indices`, for the labels. 
        labels_mask_indices = ~inputs_mask_indices

        # Using the `inputs_mask_indices`, inputs_ids at masking indices are replaced by a sentinel_id. 
        # If consecutive input_ids need to be masked, they are all replaced by one sentinel_id. 
        input_ids_sentinel = self.create_sentinel_ids(inputs_mask_indices.astype(np.int8))
        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)

        # Similarly, the labels are created for each sequence in the batch, using the `labels_mask_indices`. 
        labels_sentinel = self.create_sentinel_ids(labels_mask_indices.astype(np.int8))
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        # Say our input_ids, abcdefghi, is masked to become:
        # abc<extra_id_0>ghi
        # Then our labels for the sequence is:
        # <extra_id_0>def

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        batch = {k: torch.from_numpy(v) for k, v in batch.items()}

        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids, labels: bool = False):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)

        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed

        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        # Commented out below, as adding eos token adds uneeded noise imo. 
        # input_ids = np.concatenate(
        #     [
        #         input_ids,
        #         np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32),
        #     ],
        #     axis=-1,
        # )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans
        )

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]


@dataclass
class DataCollatorForT5MLMBloomberg:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `input_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    
    Attributes
    ----------
    tokenizer : transformers.AutoTokenizer
        The tokenizer used for encoding the data.
    noise_density : float
        The probability with which to (randomly) mask tokens in the input.
    mean_noise_span_length : float
        The average span length of the masked tokens.
    input_length : int
        The expected input length after masking.
    target_length : int
        The expected target length after masking.
    """
    tokenizer: AutoTokenizer
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:

        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )

        # Get the input_ids, before masking, for each input sequence in the batch.
        input_ids = batch["input_ids"]

        # Get the batch size and the size of the input_ids, before masking, for each input seqence in the batch.
        batch_size, expandend_input_length = input_ids.shape

        # For each input sequence in the batch, get a boolean array indicating whether to mask or not. 
        # Masking for each input sequence is determined by the `random_spans_noise_mask` method. 
        mask_booleans = [self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)]

        # As random_spans_noise_mask always masks same token numbers, if counting from right. 
        # mask_booleans = shift_list_elements(mask_booleans, count_consecutive_groups(mask_booleans))

        inputs_mask_indices = np.asarray(mask_booleans)

        # For each input sequence, get a boolean array, opposite to the `inputs_mask_indices`, for the labels. 
        labels_mask_indices = ~inputs_mask_indices

        # Using the `inputs_mask_indices`, inputs_ids at masking indices are replaced by a sentinel_id. 
        # If consecutive input_ids need to be masked, they are all replaced by one sentinel_id. 
        input_ids_sentinel = self.create_sentinel_ids(inputs_mask_indices.astype(np.int8))
        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)

        # Similarly, the labels are created for each sequence in the batch, using the `labels_mask_indices`. 
        labels_sentinel = self.create_sentinel_ids(labels_mask_indices.astype(np.int8))
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        # Say our input_ids, abcdefghi, is masked to become:
        # abc<extra_id_0>ghi
        # Then our labels for the sequence is:
        # <extra_id_0>def

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        batch = {k: torch.from_numpy(v) for k, v in batch.items()}

        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids, labels: bool = False):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)

        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed

        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32),
            ],
            axis=-1,
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans
        )

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]
    

@dataclass
class DataCollatorForT5UL2Bloomberg(DataCollatorMixin):
    """

    Data collator used for UL2

    """
    tokenizer: PreTrainedTokenizerBase
    r_denoising: bool = True
    r_probability: float = 0
    r_denoising_config: Tuple[Tuple] = ((3, 0.15),)
    s_denoising: bool = True
    s_probability: float = 0
    x_denoising: bool = True
    x_probability: float = 1
    x_denoising_config: Tuple[Tuple] = ((3, 0.5),)
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    label_pad_token_id: int = -100

    def __post_init__(self):
        self.total_task = [0, 1, 2]
        task_prob = []
        task_prob.append(self.r_probability if self.r_denoising else 0.0)
        task_prob.append(self.s_probability if self.s_denoising else 0.0)
        task_prob.append(self.x_probability if self.x_denoising else 0.0)
        self.task_prob = task_prob
        self.pad_token_id = self.tokenizer.pad_token_id
        self.decoder_start_token_id = self.tokenizer.bos_token_id

    def assign_task_type(self, batch_size: int):
        '''
        Randomly assign S,R,X to each sentence based on weighted prob
        '''
        return random.choices(self.total_task,weights=self.task_prob, k=batch_size)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        # print(examples)

        # Assign batches to different denoising tasks. 
        task_ids = self.assign_task_type(len(examples))
        task_type = torch.tensor(task_ids)
        # Get lengths of each input batch. 
        lengths = torch.tensor([ len(e['input_ids']) for e in examples ], dtype=torch.long)
        
        # If specified, pad to a multiple of self.pad_to_multiple_of. 
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer,
                    pad_to_multiple_of=self.pad_to_multiple_of)
            }
        # Get (updated) length of each input batch. 
        max_length = batch['input_ids'].shape[-1]

        # Create batches of zeroes, to fill in based on denoising task.
        new_batch = {
            "input_ids": torch.zeros(batch['input_ids'].shape, dtype=torch.long),
            "labels": torch.zeros(batch['input_ids'].shape, dtype=torch.long)
        }

        _, expanded_length = batch['input_ids'].shape
        input_ids = batch["input_ids"]

        # For batches assigned R-Denoising, create corresponding inputs and labels. 
        r_denoising_idx = task_type == 0
        if r_denoising_idx.any():
            mask_indices = None
            sub_input_ids = input_ids[r_denoising_idx]
            # union of different denoising settings
            for (mean_span, noise) in self.r_denoising_config:
                _mask_indices = np.array([
                    random_spans_noise_mask(expanded_length, mean_span, noise) for _ in range(len(sub_input_ids))
                ])
                if mask_indices is None:
                    mask_indices = _mask_indices
                else:
                    mask_indices = mask_indices | _mask_indices
            
            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
            labels_mask = ~mask_indices
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
            _sub_input_ids = self.filter_input_ids(sub_input_ids, input_ids_sentinel)
            _labels = self.filter_input_ids(sub_input_ids, labels_sentinel)
            diff = max_length-_labels.shape[-1]
            _labels = np.pad(_labels, [(0,0), (0, diff)], 'constant')
            diff = max_length - _sub_input_ids.shape[-1]
            _sub_input_ids = np.pad(_sub_input_ids, [(0,0), (0, diff)], 'constant')
            new_batch['input_ids'][r_denoising_idx] = torch.from_numpy(_sub_input_ids).long()
            new_batch['labels'][r_denoising_idx] = torch.from_numpy(_labels).long()

        # For batches assigned S-Denoising, create corresponding inputs and labels.
        s_denoising_idx = task_type == 1
        if s_denoising_idx.any():
            sub_input_ids = input_ids[s_denoising_idx]
            _labels = []
            _input_ids = []
            for input_id, len_ in zip(sub_input_ids, lengths[s_denoising_idx]):
                split = max(len_//2, 2)
                diff = expanded_length - split
                _input_ids.append(F.pad(input_id[:split], (0, diff), 'constant', self.pad_token_id))
                past_seq = input_id[split:]
                if past_seq[-1] != self.tokenizer.eos_token_id:
                    past_seq[-1] = self.tokenizer.eos_token_id
                _labels.append(F.pad(past_seq, (0, split), 'constant', self.pad_token_id))

            new_batch['input_ids'][s_denoising_idx] = torch.stack(_input_ids)
            new_batch['labels'][s_denoising_idx] = torch.stack(_labels)

        # For batches assigned X-Denoising, create corresponding inputs and labels.
        x_denoising_idx = task_type == 2
        if x_denoising_idx.any():
            mask_indices = None
            sub_input_ids = input_ids[x_denoising_idx]
            for (mean_span, noise) in self.x_denoising_config:
                _mask_indices = np.array([
                    random_spans_noise_mask(expanded_length, mean_span, noise) for _ in range(len(sub_input_ids))
                ])
                if mask_indices is None:
                    mask_indices = _mask_indices
                else:
                    mask_indices = mask_indices | _mask_indices

            labels_mask = ~mask_indices
            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
            _sub_input_ids = self.filter_input_ids(sub_input_ids, input_ids_sentinel)
            _labels = self.filter_input_ids(sub_input_ids, labels_sentinel)
            diff = max_length-_labels.shape[-1]
            _labels = np.pad(_labels, [(0,0), (0, diff)], 'constant')
            diff = max_length - _sub_input_ids.shape[-1]
            _sub_input_ids = np.pad(_sub_input_ids, [(0,0), (0, diff)], 'constant')
            new_batch['input_ids'][x_denoising_idx] = torch.from_numpy(_sub_input_ids).long()
            new_batch['labels'][x_denoising_idx] = torch.from_numpy(_labels).long()

        return self.prepare_inputs_and_labels(new_batch)


    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = []
        for row in input_ids_full:
            collapsed_id = row[row >= 0]
            diff = len(row) - len(collapsed_id)
            collapsed_id = np.pad(collapsed_id, (0, diff), 'constant')
            input_ids.append(collapsed_id)
        return np.array(input_ids)

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids
    
    def prepare_inputs_and_labels(self, batch):
        """
        TBC
        """

        # Set padding tokens to -100 for labels. 
        batch["labels"][ batch["labels"] == self.pad_token_id ] = self.label_pad_token_id
    
        return batch
