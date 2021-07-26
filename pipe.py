from fastNLP.io import Pipe, DataBundle,Loader
from fastNLP import DataSet, Instance
from transformers import BartTokenizer
import json
import numpy as np
from itertools import chain
from functools import cmp_to_key

def cmp_aspect(v1, v2):
    if v1[0]['from']==v2[0]['from']:
        return v1[1]['from'] - v2[1]['from']
    return v1[0]['from'] - v2[0]['from']

class BartABSAPipe(Pipe):
    def __init__(self, tokenizer='facebook/bart-base', opinion_first=True):
        super(BartABSAPipe, self).__init__()
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer)
        self.mapping = {  # so that the label word can be initialized in a better embedding.
            'POS': '<<positive>>',
            'NEG': '<<negative>>',
            'NEU': '<<neutral>>'
        }
        for tok in self.mapping.values():
            assert self.tokenizer.convert_tokens_to_ids([tok])[0] == self.tokenizer.unk_token_id
        self.tokenizer.add_tokens(list(self.mapping.values()), special_tokens=True)

        self.mapping2id = {}
        self.mapping2targetid = {}
        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= self.tokenizer.vocab_size
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid)

    def process(self, data_bundle: DataBundle) -> DataBundle:
        """
        :param data_bundle: 读入的原始数据
        :return:
            tgt_ids：目标 tokens 序列 -> [<s>, span1, span2,..., spanN, </s>]
                     其中 spani = a_s, a_e, o_s, o_e, c
                     并且 target id 用的是自定义的 map 中的下标
                        -> 自定义 map 中元素为 [<s>,</s>,labels,src_tokens]
                                              0    1   2,3,4    5~N
        """
        target_shift = len(self.mapping) + 2
                # src_tokens 对应的 position index 应该向后推 num of (labels, <s> and </s>)

        def prepare_target(ins):
            raw_words = ins['raw_words']
            # 准备阶段
            whole_word_ids = [[self.tokenizer.bos_token_id]]
            for i, ele in enumerate(raw_words):
                if i:
                    whole_word_ids.append(self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(ele, add_prefix_space=True)))
                else:
                    whole_word_ids.append(self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(ele)))
            whole_word_ids.append([self.tokenizer.eos_token_id])
            lens = list(map(len, whole_word_ids))       # 统计各个 unit-words 的 length
            cum_lens = np.cumsum(list(lens)).tolist()   # 代表着后一位的下标

            # 开始构建 tgt_ids
            tgt_ids = [0]          # 0 为自定义 map 中 <s> 下标
            target_spans = []
            src_ids = list(chain(*whole_word_ids))
            # 1_处理 aspect term 和 opinion term
            # 正序排列 aspects_opinions
            aspects_opinions = [(a, o) for a, o in zip(ins['aspects'], ins['opinions'])]
            aspects_opinions = sorted(aspects_opinions, key=cmp_to_key(cmp_aspect))
            # 搞定 as,ae,os,oe 部分
            for aspects, opinions in aspects_opinions:  # 预测bpe的start
                assert aspects['index'] == opinions['index']
                a_start = cum_lens[aspects['from']]
                    # 因为 cum_lens 中多出 <s>，所以 "from" 为 token 前一位置
                    # -> 正好前一位置的存储的是前面 seq 的长度 == as 指代的下标
                a_end = cum_lens[aspects['to']-1]    # 开区间 -> 闭区间
                o_start = cum_lens[opinions['from']]
                o_end = cum_lens[opinions['to']-1]
                # 这里需要evaluate是否是对齐的
                # for idx, word in zip((o_start_bpe, o_end_bpe, a_start_bpe, a_end_bpe),
                #                      (opinions['term'][0], opinions['term'][-1], aspects['term'][0], aspects['term'][-1])):
                #     assert _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[:1])[0] or \
                #            _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[-1:])[0]

                target_spans.append([a_start+target_shift, a_end+target_shift,
                                     o_start+target_shift, o_end+target_shift])
                target_spans[-1].append(self.mapping2targetid[aspects['polarity']]+2)   # 前面有sos和eos
                target_spans[-1] = tuple(target_spans[-1])
            tgt_ids.extend(list(chain(*target_spans)))
            tgt_ids.append(1)           # 1 为自定义 map 中的 </s>
            return {'tgt_tokens': tgt_ids, 'target_span': target_spans, 'src_tokens': src_ids}
                    # tgt_ids 是 map 中的 index；src_ids 是 vocab 中的 index

        data_bundle.apply_more(prepare_target, use_tqdm=True, tqdm_desc='Pre. tgt.')

        data_bundle.set_ignore_type('target_span')
        data_bundle.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'target_span')
        return data_bundle

    def process_from_file(self, paths) -> DataBundle:
        data_bundle = BartABSALoader().load(paths)
        print(data_bundle)
        data_bundle = self.process(data_bundle)
        print(data_bundle)
        return data_bundle

class BartABSALoader(Loader):
    def __init__(self):
        super().__init__()

    def _load(self, path: str) -> DataSet:
        with open(path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        ds = DataSet()
        for ins in data:
            tokens = ins['words']
            aspects = ins['aspects']
            opinions = ins['opinions']
            assert len(aspects)==len(opinions)
            ins = Instance(raw_words=tokens, aspects=aspects, opinions=opinions)
            ds.append(ins)
        return ds

if __name__ == "__main__":
    data_bundle = BartABSALoader().load(paths="./data/D20b/14lap/")
    print(data_bundle.get_dataset("train"))