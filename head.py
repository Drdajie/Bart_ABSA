import torch
from transformers import BartTokenizer
import utilities
from itertools import chain

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = "facebook/bart-base"
model_path = "./model_parameters/model_state"
bos_token_id, eos_token_id = 0, 1       # 自定义 mapping 中 <s> </s> 的位置


#gpus = [3,1]