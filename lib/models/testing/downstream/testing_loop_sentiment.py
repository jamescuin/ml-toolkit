###################### Imports #################################
import logging
import torch
from tqdm.auto import tqdm
import numpy as np
from datetime import datetime as dt
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from lib.models.handlers.model_loader import ModelLoader
from lib.models.handlers.adapter_handler import AdapterHandler
from typing import Dict
from peft import PeftModel


###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)

###################### Functions #################################


def test_model_sentiment(
        model,
        tokenizer,
        device,
        test_dataloader,
        show_n_predictions: int = 0,
        show_progress_bars: bool = False,
        max_target_length: int = None,
):
    """
    Logs the accuracy of the model on the test dataset. If specified, `show_n_predictions`
    example predictions are also logged. 
    """
    if max_target_length is None: 
        max_target_length = tokenizer.model_max_length

     # Now evaluate on the test set
    # This dictionary provides a mapping from the text labels to integer labels for metric calculation
    TEXT_TO_ID = {'negative': 0,
                'positive': 1,
                'neutral': 2}
    ID_TO_TEXT = {TEXT_TO_ID[k]: k for k in TEXT_TO_ID}

    # These are for tracking performance
    test_accuracies = []

    test_inputs_text = []
    test_predictions_text = []
    test_predictions = []
    test_labels = []

    if show_progress_bars:
        test_progress_bar = tqdm(range(len(test_dataloader)))

    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            # Pass the inputs to GPU
            model_batch = {k: v.to(device) for k, v in batch.items() if k != 'labels'}

            # Generate text given input
            outputs = model.generate(
                **model_batch,
                max_length=max_target_length,
            )

            batch['labels'] = torch.where(batch['labels'] == torch.tensor(-100), torch.tensor(tokenizer.pad_token_id), batch['labels'])
            
            input_text = tokenizer.batch_decode(
                model_batch['input_ids'],
                skip_special_tokens=False
            )

            # Decode the model predictions back to text
            pred_text = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            
            label_text = tokenizer.batch_decode(
                batch['labels'].detach().cpu().numpy(),
                skip_special_tokens=True
            )

            # Convert the text to numerical
            # We use the get() method because at the start T5 might produce text that is not quite right
            # and which might not map to any particular label, so we default to negative if doesn't predict
            y_true = np.array([TEXT_TO_ID.get(text, 0) for text in label_text])
            preds = np.array([TEXT_TO_ID.get(text, 0) for text in pred_text])

            test_acc = np.mean(y_true == preds)
            test_accuracies.append(test_acc)
            test_mean_acc = np.mean(test_accuracies)

            # Keep track of predictions for end of epoch calculations
            test_predictions.append(preds)
            test_labels.append(y_true)
            test_predictions_text.append(pred_text)
            test_inputs_text.append(input_text)

            if show_progress_bars:
                test_progress_bar.update(1)
                test_progress_bar.set_postfix(acc=test_mean_acc)

    test_labels = np.concatenate(test_labels, axis=0)
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_predictions_text = np.concatenate(test_predictions_text, axis=0)
    test_inputs_text = np.concatenate(test_inputs_text, axis=0)

    test_prec, test_recall, test_f1, _ = precision_recall_fscore_support(
        test_labels,
        test_predictions,
        average='macro'
    )

    log.info(f'Test F1: {test_f1}, Precision: {test_prec}, Recall: {test_recall}, Accuracy: {test_mean_acc}')

    # Get per class breakdown
    test_prec_br, test_recall_br, test_f1_br, _ = precision_recall_fscore_support(
        test_labels,
        test_predictions,
        average=None,
        zero_division=0
    )

    for i in range(3):
        log.info(f'Test performance on {ID_TO_TEXT[i]} class')
        log.info(f'\t F1: {test_f1_br[i]}, Precision: {test_prec_br[i]}, Recall: {test_recall_br[i]}')

    if show_n_predictions > 0: 
        log.info(f'Printing {show_n_predictions} predictions...')
        for i in range(show_n_predictions):
            log.info('Input text')
            log.info(f'\t {test_inputs_text[i]}')
            
            log.info('Target')
            log.info(f'\t {test_labels[i]}')
            
            log.info('Prediction')
            log.info(f'\t {test_predictions[i]}')

            log.info('Prediction Text')
            log.info(f'\t {test_predictions_text[i]}')
            
            log.info('######')
        

    return test_acc, test_prec, test_recall, test_f1


def test_model_sentiment_kfold(
        dataloaders_kfold,
        pretrained_model_type: str,
        pretrained_model_dir: str,
        best_model_directory_path: str,
        device,
        adapter_type: str,
        show_n_predictions: int,
        show_progress_bars: bool = False,
        max_target_length: int = None,
        using_adapter: bool = True,
):
    """
    TBC
    """
    test_results = {}

    for fold, fold_dataloader in dataloaders_kfold.items():
        fold_best_model_directory_path = best_model_directory_path + f'/{fold}'

        # Model & Tokenizer Loading
        if using_adapter:
            model_loader = ModelLoader(
                model_path=pretrained_model_dir,
                model_type=pretrained_model_type,
                device=device,
                )
            tokenizer = model_loader.load_tokenizer()
            fold_model = model_loader.load_model()
            adapter_handler = AdapterHandler(
                model=fold_model, 
                adapter_type=adapter_type,
                device=device
                )
            fold_model = adapter_handler.merge_adapter_into_model(fold_best_model_directory_path)

        else:
            model_loader = ModelLoader(
                model_path=fold_best_model_directory_path,
                model_type=pretrained_model_type,
                device=device,
                )
            tokenizer = model_loader.load_tokenizer()
            fold_model = model_loader.load_model()

        fold_test_results = test_model_sentiment(
            model=fold_model,
            tokenizer=tokenizer,
            device=device,
            test_dataloader=fold_dataloader['test'],
            show_n_predictions=show_n_predictions,
            show_progress_bars=show_progress_bars,
            max_target_length=max_target_length,
        )
        test_results[fold] = fold_test_results
        log.info(fold_test_results)

    return test_results

def calculate_test_results_across_folds(test_results_per_fold: Dict[str, tuple]):
    """
    TBC
    """
    acc_each_fold = []
    prec_each_fold = []
    recall_each_fold = []
    f1_each_fold = []

    for fold, fold_test_results in test_results_per_fold.items():
        acc_each_fold.append(np.mean(fold_test_results[0]))
        prec_each_fold.append(np.mean(fold_test_results[1]))
        recall_each_fold.append(np.mean(fold_test_results[2]))
        f1_each_fold.append(np.mean(fold_test_results[3]))

    avg_acc = np.mean(acc_each_fold)
    avg_prec = np.mean(prec_each_fold)
    avg_recall = np.mean(recall_each_fold)
    avg_f1 = np.mean(f1_each_fold)

    std_acc = np.std(acc_each_fold)
    std_prec = np.std(prec_each_fold)
    std_recall = np.std(recall_each_fold)
    std_f1 = np.std(f1_each_fold)

    log.info('Test Results Across Folds:')

    log.info(f'Mean Acc: {avg_acc}')
    log.info(f'Mean Prec: {avg_prec}')
    log.info(f'Mean Recall: {avg_recall}')
    log.info(f'Mean F1: {avg_f1}')

    log.info(f'Std Prec: {std_acc}')
    log.info(f'Std Prec: {std_prec}')
    log.info(f'Std Recall: {std_recall}')
    log.info(f'Std F1: {std_f1}')

    return avg_acc, avg_prec, avg_recall, avg_f1, std_acc, std_prec, std_recall, std_f1
