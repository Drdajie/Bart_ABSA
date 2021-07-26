import json
from torch.utils.data import Dataset, DataLoader
import head
import utilities
from functools import cmp_to_key
import numpy as np
from itertools import chain
import torch

def prep_dataloader(data_prefix, data_name, mode, batch_size,  max_seq_len):
    """
    得到相应的 DataLoader
    :param data_prefix: 选用哪个数据集的前缀，可选 ["D17"、"D19"、"D20a"、"D20b"]
    :param data_name: 选用哪个数据集，可选 ["14lap"、"14res"、"15res"、"16res"]
    :param mode: 可选 ["train"、"dev"、"test"] 三种
    :param batch_size:
    :param max_seq_len:
    :return:
    """
    dataset_dict = {"D20b": D20bDataset}
    dataset = dataset_dict[data_prefix](mode=mode, data_name=data_name, max_seq_length = max_seq_len)
    dataLoader = DataLoader(dataset, batch_size, shuffle=(mode=="train"))
    return dataset, dataLoader



def convert_examples_to_features(datas, max_len,tokenizer = head.tokenizer):
    results = []
    def get_one_res(ins):
        # 1_预处理（处理拼接单词的情况）
        raw_words = ins['raw_words']
        token_ids_lists = [[tokenizer.bos_token_id]]
        for i,word in enumerate(raw_words):
            if i:
                token_ids_lists.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word, add_prefix_space=True)))
            else:
                token_ids_lists.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)))
        token_ids_lists.append([tokenizer.eos_token_id])

        lens = list(map(len, token_ids_lists))  # 统计各个 unit-words 的 length
        cum_lens = np.cumsum(list(lens)).tolist()  # 代表着后一位的下标
        # 2_开始构造 target_seq
        target_seq = []                # 最后要构造的玩意儿
        target_shift = len(utilities.label_map) + 1  # 最后要进行 softmax 时的表中要手动加入的 labels 和 </s> 个数
        target_spans = []
        token_ids = list(chain(*token_ids_lists))
        mask_ids = [1 if i < cum_lens[-1] else 0 for i in range(max_len)]

        aspects_opinions = [(a, o) for a, o in zip(ins['aspects'], ins['opinions'])]
        aspects_opinions = sorted(aspects_opinions, key=cmp_to_key(utilities.cmp_aspect))
            # 正序排列 aspects_opinions

        for aspects, opinions in aspects_opinions:  # 预测bpe的start
            assert aspects['index'] == opinions['index']
            a_start = cum_lens[aspects['from']]  # 因为有一个sos shift
            a_end = cum_lens[aspects['to'] - 1]  # 这里由于之前是开区间，刚好取到最后一个word的开头
            o_start = cum_lens[opinions['from']]  # 因为有一个sos shift
            o_end = cum_lens[opinions['to'] - 1]  # 因为有一个sos shift
            # 这里需要evaluate是否是对齐的
            # for idx, word in zip((o_start, o_end, a_start, a_end),
            #                      (opinions['term'][0], opinions['term'][-1], aspects['term'][0], aspects['term'][-1])):
            #     assert token_ids[idx] == \
            #            self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[:1])[
            #                0] or \
            #            _word_bpes[idx] == \
            #            self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[-1:])[
            #                0]
            target_spans.append([a_start + target_shift, a_end + target_shift,
                                 o_start + target_shift, o_end + target_shift])
            target_spans[-1].append(utilities.label2id_map[aspects['polarity']])
            target_spans[-1] = tuple(target_spans[-1])
        target_seq.extend(list(chain(*target_spans)))
        target_seq.append(0)        # 0 代表生成一个句子时用的 vocab 里的 </s>
        tgt_len = len(target_seq)
        tgt_mask = torch.tensor([False if i < tgt_len else True for i in range(max_len)])
        target_seq_temp = torch.tensor(target_seq)
        target_seq = torch.empty(max_len)
        target_seq[:tgt_len] = target_seq_temp
        target_seq.masked_fill_(tgt_mask, -100)
        target_seq = target_seq.long()
        # 转类型
        token_ids = utilities.padding(token_ids, max_len, 1)
        token_ids = token_ids.long()
        mask_ids = torch.tensor(mask_ids, dtype=torch.long)
        return {"raw_words":raw_words, "src_token_ids":token_ids, "mask_ids":mask_ids, "tgt_seq":target_seq}
    for item in datas:
       results.append(get_one_res(item))
    return results

class D20bDataset(Dataset):
    def __init__(self, mode, data_name, max_seq_length):
        super(D20bDataset, self).__init__()
        file_path = "./data/D20b/{}/{}_convert.json".format(data_name, mode)
        with open(file_path, "r") as f:
            data_all = json.load(f)
        self.datas = convert_examples_to_features(data_all, max_len=max_seq_length)

    def __getitem__(self, item):
        return  self.datas[item]

    def __len__(self):
        return len(self.datas)

class D17Dataset(Dataset):
    def __init__(self, mode, data_name, max_seq_length):
        super(D17Dataset, self).__init__()
        file_path = "./data/D_17/{}/{}_convert.json".format(data_name, mode)
        with open(file_path, "r") as f:
            data_all = json.load(f)
        self.seqs, self.labels = self.process_data(data_all)

    def process_data(self, data):
        seqs, labels = [], []
        for item in data:
            words = item["raw_words"]
            aspect = item["aspects"]
            opinions = item["opinions"]
            seqs.append(words)
            label = []
            for ai, oi in zip(aspect, opinions):
                if "polarity" not in ai:
                   continue
                labels.append(ai["from"])
                labels.append(ai["to"])
                labels.append(oi["from"])
                labels.append(oi["to"])
                labels.append(ai["polarity"])
            labels.append(label)
        return seqs, labels

    def __getitem__(self, item):
        return self.seqs[item], self.labels[item]

    def __len__(self):
        return len(self.seqs)


if __name__ == "__main__":
    dataset = D17Dataset(mode = "train", data_name="15res", max_seq_length=0)
    print(len(dataset))