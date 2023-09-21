###################### Imports #################################
import logging
import os
import torch
from tqdm.auto import tqdm
import numpy as np
from rouge import Rouge
from lib.models.handlers.model_loader import ModelLoader
from lib.models.handlers.adapter_handler import AdapterHandler

###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)

###################### Functions #################################


def test_model_summarization(
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

    # These are for tracking performance
    test_rouge1_f = []
    test_rouge1_p = []
    test_rouge1_r = []

    predictions = []
    labels = []
    inputs = []

    rouge = Rouge()

    if show_progress_bars:
        progress_bar = tqdm(range(len(test_dataloader)))

    with torch.no_grad():

        for idx, batch in enumerate(test_dataloader):
            # Pass the inputs to GPU
            model_batch = {k: v.to(device) for k, v in batch.items() if k != 'labels'}

            batch['labels'] = torch.where(batch['labels'] == torch.tensor(-100), torch.tensor(tokenizer.pad_token_id), batch['labels'])

            # Generate text given input
            outputs = model.generate(
                **model_batch,
                max_length=max_target_length,
                repetition_penalty=2.5
            )
            
            inputs += tokenizer.batch_decode(
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

            # Process predictions and labels 
            pred_text = [p if p not in ['.', ''] else '/' for p in pred_text]

            test_rouge_scores = rouge.get_scores(pred_text, label_text, avg=True)

            test_rouge1_f.append(test_rouge_scores['rouge-1']['f'])
            test_rouge1_p.append(test_rouge_scores['rouge-1']['p'])
            test_rouge1_r.append(test_rouge_scores['rouge-1']['r'])

            test_mean_rouge1_f = np.mean(test_rouge1_f)
            test_mean_rouge1_p = np.mean(test_rouge1_p)
            test_mean_rouge1_r = np.mean(test_rouge1_r)

            labels += label_text
            predictions += pred_text

            if show_progress_bars:
                progress_bar.update(1)
                progress_bar.set_postfix(rouge_1_f=test_mean_rouge1_f)

    log.info(f'Test ROUGE-1 F: {test_mean_rouge1_f}, Test ROUGE-1 Precision: {test_mean_rouge1_p}, Test ROUGE-1 Recall: {test_mean_rouge1_r}')

    if show_n_predictions > 0: 
        log.info(f'Printing {show_n_predictions} predictions...')
        for i in range(show_n_predictions):
            log.info('Input text')
            log.info(f'\t {inputs[i]}')
            
            log.info('Target')
            log.info(f'\t {labels[i]}')
            
            log.info('Prediction')
            log.info(f'\t {predictions[i]}')
            
            log.info('######')
        

    return test_mean_rouge1_p, test_mean_rouge1_r, test_mean_rouge1_f


def test_model_summarization_kfold(
        dataloaders_kfold,
        pretrained_model_dir: str,
        pretrained_model_type: str,
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

        fold_test_results = test_model_summarization(
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
