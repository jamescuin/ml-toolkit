###################### Imports #################################

import logging
import torch
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel
from lib.models.model_utils import get_trainable_parameters
from lib.utils import get_from_map

###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


###################### Classes #################################

class AdapterHandler:
    """
    A handler for managing different types of adapters in models.

    Parameters
    ----------
    model : object
        The model to be managed.
    adapter_type : str
        The type of the adapter. Must be one of the supported adapter types ('lora', etc.)
    device : str, optional
        The device to use for computations ('cuda', 'cpu', etc.). Defaults to 'cuda' if available.
    """
    def __init__(self, model, adapter_type: str, device: str = None):
        self.model = model
        self.adapter_type = adapter_type
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.validate_adapter_type()

    def validate_adapter_type(self):
        """Checks if the provided adapter_type is supported."""
        supported_adapters = ['lora']
        if self.adapter_type not in supported_adapters:
            raise ValueError(f"Unsupported adapter type: {self.adapter_type}")

    def add_adapter_to_model(self, **kwargs):
        """
        Adds a specific adapter to the `self.model` based on `self.adapter_type`.
        """
        adapter_map = {
            'lora': self.add_lora_adapter_to_model
        }

        peft_model = get_from_map(self.adapter_type, adapter_map, **kwargs)

        return peft_model
    
    def add_lora_adapter_to_model(self, r: int = 32, lora_alpha: int = 32, lora_dropout: float = 0.1):
        """
        Adds a LoRa adapter to the `self.model`. 
        """
        log.info('Using LoRA adapter...')
        peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        peft_model = get_peft_model(self.model, peft_config)
        peft_model.to(self.device)

        log.info(f'Adapter trainable parameters are:\n{get_trainable_parameters(peft_model)}')
        log.info(f'PEFT Config is:\n{peft_model.peft_config}')

        return peft_model
    
    def merge_adapter_into_model(self, adapter_dir_path: str):
        """
        Merges the adapter into the `self.model`.
        """
        model_to_merge = PeftModel.from_pretrained(self.model, adapter_dir_path)
        merged_model = model_to_merge.merge_and_unload()
        return merged_model
