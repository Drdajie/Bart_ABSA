# -*- encoding:utf-8 -*-
import torch

a = torch.arange(5)
b = a.expand(2,-1)
c = torch.LongTensor([2,3]).reshape(-1,1)
print(b.lt(c))

# import numpy as np
# import transformers
# print(torch.__version__)  # 1.7.1
# print(transformers.__version__) # 2.1.1
# import torch.nn as nn
# import torch.nn.functional as F
#
# a = torch.randn([2,4,3], requires_grad=True)
# b = torch.tensor([[0,0,0],[0,0,0]], dtype=torch.long)
# losser = nn.CrossEntropyLoss()
# loss = losser(a,b)
# # loss = F.cross_entropy(target=b, input=a)
# print(loss)

# from transformers import BartTokenizer, BartModel
# from itertools import chain
# import torch

# model_name = "facebook/bart-base"
# tokenizer = BartTokenizer.from_pretrained(model_name)
# model = BartModel.from_pretrained(model_name)
# print(id(model.encoder.embed_tokens))
# print(id(model.decoder.embed_tokens))
# print(id(model.encoder.embed_tokens.weight))
# inputs = tokenizer("I love you", return_tensors="pt")
# outputs = model(**inputs, return_dict=True, output_hidden_states=True)
# inputs_embeds = model.encoder.embed_tokens(inputs.input_ids) * model.encoder.embed_scale
# embed_pos = model.encoder.embed_positions(inputs.input_ids)
# x = inputs_embeds + embed_pos
# print(outputs.encoder_hidden_states[4] - x)
# # print(outputs.encoder_hidden_states[2].shape)
# # print(outputs.encoder_hidden_states[0].shape)
# print(outputs.encoder_last_hidden_state)

# with open("./data/D_20b/16res/train_convert.json", "r") as f:
#
#     ids = [[tokenizer.bos_token_id]]
#     for i,ele in enumerate(words):
#         if i:   ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ele, add_prefix_space = True)))
#         else:   ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ele)))
#     for ele in words:
#         ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ele, add_prefix_space = True)))
#     ids.append([tokenizer.eos_token_id])
#     print(ids)
# lens = list(map(len,ids))
# cum_lens = np.cumsum(list(lens)).tolist()
# print(lens)
# print("cum_lens：", cum_lens)
# a_start_bpe = cum_lens[8]
# a_end_bpe = cum_lens[13]
# print(a_start_bpe, a_end_bpe-1)
# ids = list(chain(*ids))
# print(ids)
