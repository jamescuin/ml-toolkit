###################### Imports #################################

import logging
from torch.optim import AdamW
from transformers import get_scheduler
from lib.utils import get_from_map


###################### Variables #################################

logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


###################### Classes #################################

class ModelOptimization:
    def __init__(self, model, optimizer_type: str, scheduler_type: str):
        self.model = model
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type

    def get_optimizer(self, **kwargs):
        log.info("Getting optimizer...")
        optimizer_map = {
            'adam_w': self.get_optimizer_adamw
        }
        return get_from_map(self.optimizer_type, optimizer_map, **kwargs)

    def get_optimizer_adamw(self, learning_rate: float, weight_decay: float):
        log.info('Using AdamW optimizer...')
        return AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
    
    def get_scheduler(self, **kwargs):
        log.info("Getting scheduler...")
        scheduler_map ={
            'linear': self.get_scheduler_linear
        }
        return get_from_map(self.scheduler_type, scheduler_map, **kwargs)

    def get_scheduler_linear(self, optimizer, num_warmup_steps: int, num_training_steps: int):
        log.info("Using Linear scheduler...")
        return get_scheduler(
                name="linear", 
                optimizer=optimizer, 
                num_warmup_steps=num_warmup_steps, 
                num_training_steps=num_training_steps,
            )
    