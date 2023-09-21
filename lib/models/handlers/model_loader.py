
###################### Imports #################################

import logging
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from lib.utils import get_from_map


###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


###################### Classes #################################

class ModelLoader:
    """
    Handles loading of models and tokenizers based on type and path.

    Attributes
    ----------
    model_path : str
        Path to the model to load.
    model_type : str
        Type of the model. Currently only 't5' is supported.
    device : torch.device
        Device to load the model on. This is GPU if available. 
    tokenizer : transformers.PreTrainedTokenizer
        Loaded tokenizer.
    model : transformers.PreTrainedModel
        Loaded model.
    """
    def __init__(self, model_path: str, model_type: str, device: str = None):
        self.model_path = model_path
        self.model_type = model_type
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load_tokenizer(self, **kwargs):
        """
        Loads and returns the tokenizer based on `model_type`.
        """
        log.info(f'Loading tokenizer from: {self.model_path}...')
        tokenizer_map = {
            't5': self._load_tokenizer_t5,
            't5_flax': self._load_tokenizer_t5_flax,
            'llama2': self._load_tokenizer_llama2,
        }
        return get_from_map(self.model_type, tokenizer_map, **kwargs)

    def _load_tokenizer_t5(self):
        """
        Loads and returns the tokenizer for the t5-type model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self.tokenizer
    
    def _load_tokenizer_t5_flax(self):
        """
        Loads and returns the tokenizer for the t5-type flax model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, from_flax=True)
        return self.tokenizer
    
    def _load_tokenizer_llama2(self):
        """
        Loads and returns the tokenizer for the Llama2-type model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
        return self.tokenizer
    
    def load_model(self, **kwargs):
        """
        Loads and returns the model based on `model_type`.
        """
        log.info(f'Loading model from: {self.model_path}...')
        model_map = {
            't5': self._load_model_t5,
            't5_flax': self._load_model_t5_flax,
            'llama2': self._load_model_llama2,
        }
        return get_from_map(self.model_type, model_map, **kwargs)
    
    def _load_model_t5(self):
        """
        Loads and returns the T5-type model.
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
            device_map={'': 0},
            torch_dtype=torch.float32,
        )
        self.model.to(self.device)

        return self.model

    def _load_model_t5_flax(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
            # device_map={'': 0},  # Causes unbound error
            torch_dtype=torch.float32,
            from_flax=True
        )

        self.model.to(self.device)
        return self.model
    
    def _load_model_llama2(self):
        """
        Loads and returns the Llama2-type model.
        """
        # Activate 4-bit precision base model loading
        use_4bit = True
        # Compute dtype for 4-bit base models
        bnb_4bit_compute_dtype = "float16"
        # Quantization type (fp4 or nf4)
        bnb_4bit_quant_type = "nf4"
        # Activate nested quantization for 4-bit base models (double quantization)
        use_nested_quant = False
        # Load the entire model on the GPU 0
        device_map = {"": 0}
        
        # Load model with QLoRA configuration
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )

        # Check GPU compatibility with bfloat16
        if compute_dtype == torch.float16 and use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map=device_map
        )
        # self.model.config.use_cache = False 
        self.model.config.pretraining_tp = 1

        return self.model
