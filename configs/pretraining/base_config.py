############## Base Model ##############
BASE_MODEL = 'google/flan-t5-base'  # The model we wish to pretrain.
BASE_MODEL_TYPE = 't5'  # The model type, to be used in selecting the correct tokenizer, etc.  

############## Pre-Training Alias ##############
PRETRAINING_ALIAS = 'google-flan-t5-base-demo'  # Alias that our model is saved to after pretraining. 

############## Data Source ##############
DATA_SOURCE = 'bloomberg'  # Alias of data source used in pretraining. 
