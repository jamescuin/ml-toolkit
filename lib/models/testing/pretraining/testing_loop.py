###################### Imports #################################

import logging
from typing import Dict
import torch
import numpy as np


###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)

###################### Functions #################################


def test_model(
        model,
        tokenizer,
        device,
        test_dataloader,
) -> Dict[str, any]:
    """
    Test a trained `model` on a test dataset, associated with the `test_dataloader` and calculate the accuracy.

    Parameters
    ----------
    model
        The model to be tested.
    tokenizer
        The tokenizer used for the model.
    device : torch.device
        The device (CPU/GPU) on which the model will be tested.
    test_dataloader : torch.utils.data.DataLoader
        DataLoader containing the test data.

    Returns
    -------
    Dict[str, any]
        A dictionary with keys 'accuracy', 'input_text', 'target_text', 'predicted_text'. 
        'accuracy' contains the mean accuracy of the model on the test data.
        'input_text', 'target_text' and 'predicted_text' contain lists of corresponding texts.

    """
    accuracy_of_matches = []
    input_text, target_text, predicted_text = [], [], []

    with torch.no_grad():
        log.info(f'{len(test_dataloader)} batches to test.')
        for idx, model_batch in enumerate(test_dataloader):
            log.info(f'Testing batch {idx}...')

            model_batch = {k: v.to(device) for k, v in model_batch.items()}
            # Replace -100 with padding token_ids, so it can be decoded. 
            model_batch['labels'] = torch.where(model_batch['labels'] == -100, torch.tensor(tokenizer.pad_token_id).to(device), model_batch['labels'])
            outputs = model(**model_batch)

            input_text += tokenizer.batch_decode(model_batch['input_ids'], skip_special_tokens=False)
            target_text += tokenizer.batch_decode(model_batch['labels'], skip_special_tokens=False)
            
            
            predictions = torch.argmax(outputs.logits, -1).detach().cpu().numpy()
            predicted_text += tokenizer.batch_decode(predictions, skip_special_tokens=False)

            matches = (model_batch['labels'].detach().cpu().numpy() == predictions) * (predictions != 0)
            # .detach() detaches from the backward graph to avoid copying gradients.
            # .cpu() moves the data from GPU to CPU.
            # .numpy() converts torch.Tensor to np.ndarray.
            
            accuracy_of_matches.append(np.mean(matches, axis=1).mean())

        log.info(f'Accuracy of matches: {np.mean(accuracy_of_matches)}')

    return {
        'accuracy': np.mean(accuracy_of_matches),
        'input_text': input_text,
        'target_text': target_text,
        'predicted_text': predicted_text,
    }
