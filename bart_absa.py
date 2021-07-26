from fastNLP.models import Seq2SeqModel
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
from modeling_bart import BartDecoder, BartEncoder, BartModel
from transformers import BartTokenizer
import torch
from fastNLP import seq_len_to_mask
import torch.nn as nn
import torch.nn.functional as F

class BartSeq2SeqModel(Seq2SeqModel):
    @classmethod
    def build_model(cls, bart_model, tokenizer, label_ids, decoder_type=None, copy_gate=False,
                    use_encoder_mlp=False, use_recur_pos=False, tag_first=False):
        """
        构建一个 seq2seq model -> 主要是给其创建 encoder 和 decoder 部分。
        :param bart_model: 即 bart_name
        :param tokenizer:
        :param label_ids:
        :param decoder_type:
        :param copy_gate:
        :param use_encoder_mlp:
        :param use_recur_pos:
        :param tag_first:
        :return:
        """
        model = BartModel.from_pretrained(bart_model)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        #model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens)+num_tokens)
        model.resize_token_embeddings(len(tokenizer))
        encoder = model.encoder
        decoder = model.decoder

        if use_recur_pos:
            decoder.set_position_embedding(label_ids[0], tag_first)

        _tokenizer = BartTokenizer.from_pretrained(bart_model)
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':
                index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                if len(index)>1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                assert index>=num_tokens, (index, num_tokens, token)
                indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2]))
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                model.decoder.embed_tokens.weight.data[index] = embed

        encoder = FBartEncoder(encoder)
        label_ids = sorted(label_ids)
        print(decoder_type)
        print(decoder_type is None)
        if decoder_type is None:
            assert copy_gate is False
            decoder = FBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids)
        # elif decoder_type =='avg_score':
        #     decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
        #                                       use_encoder_mlp=use_encoder_mlp)
        else:
            raise RuntimeError("Unsupported feature.")
        return cls(encoder=encoder, decoder=decoder)

    def prepare_state(self, src_tokens, src_seq_len=None, first=None, tgt_seq_len=None):
        encoder_outputs, encoder_mask, hidden_states = self.encoder(src_tokens, src_seq_len)
        src_embed_outputs = hidden_states[0]
        state = BartState(encoder_outputs, encoder_mask, src_tokens, first, src_embed_outputs)
        # setattr(state, 'tgt_seq_len', tgt_seq_len)
        return state

    def forward(self, src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first):
        """

        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor first: 显示每个, bsz x max_word_len
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        state = self.prepare_state(src_tokens, src_seq_len, first, tgt_seq_len)
        decoder_output = self.decoder(tgt_tokens, state)
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': decoder_output}
        elif isinstance(decoder_output, (tuple, list)):
            return {'pred': decoder_output[0]}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")

class FBartEncoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = encoder

    def forward(self, src_tokens, src_seq_len):
        mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        dict = self.bart_encoder(input_ids=src_tokens, attention_mask=mask, return_dict=True,
                                 output_hidden_states=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        return encoder_outputs, mask, hidden_states

class FBartDecoder(Seq2SeqDecoder):
    def __init__(self, decoder, pad_token_id, label_ids, use_encoder_mlp=True):
        super().__init__()
        assert isinstance(decoder, BartDecoder)
        self.decoder = decoder
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id
        self.label_start_id = label_ids[0]
        self.label_end_id = label_ids[-1]+1
        # 0th position is <s>, 1st position is </s>
        mapping = torch.LongTensor([0, 2]+sorted(label_ids, reverse=False))
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping)  # 加上一个
        hidden_size = decoder.embed_tokens.weight.size(1)
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.3),
                                             nn.ReLU(),
                                             nn.Linear(hidden_size, hidden_size))

    def forward(self, tokens, state):
        """
        :param tokens: 即 tgt_tokens -> 在训练阶段其为输入 -> 所以也要做 padding 操作
                       例如 [[as1,ae1,os1,oe1,c1,1,...,as3,ae3,os3,oe3,c3,1],
                            [as1,ae1,os1,oe1,c1,1,1,1,...,1,1,1,1,1,1,1,1,],
                            ...,
                            [...]                                           ]
        :param state:
        :return:
        """
        # bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        first = state.first

        # eos is 1
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
                # FFFFFTTTTT -> TTTTTFFFFFF -> 123444444
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])     # not equal
                # 因为我们只要第一个 </s> 之前的，即只要 cumsum 中等于最后一位的元素
                # 444444321 -> FFFFFFTTT

        # mapping to the BART token index
        # 1_mapping labels and <s>, </s> to embedding_ids
        mapping_token_mask = tokens.lt(self.src_start_index)  # less than -> 找出 labels 和 <s></s>
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)  # greater or equal
        tag_mapped_tokens = self.mapping[mapped_tokens]  # tensor([0, 2, 50265, 50266, 50267])
        # 2_mapping src_tokens to embedding_ids
        src_tokens_index = tokens - self.src_start_index # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)   # shape equals src_tokens_index

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)

        if self.training:
            tokens = tokens[:, :-1]         # 输入不需要 tokens 的 </s>
            decoder_pad_mask = tokens.eq(self.pad_token_id)
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)

        # first get the
        eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[2:3])  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id])  # bsz x max_len x num_class

        # bsz x max_word_len x hidden_size
        src_outputs = state.encoder_output

        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)

        mask = mask.unsqueeze(1).__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        return logits

    def decode(self, tokens, state):
        return self(tokens, state)[:, -1]

class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, first, src_embed_outputs):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs, indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(layer[key1][key2], indices)
                            # print(key1, key2, layer[key1][key2].shape)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new