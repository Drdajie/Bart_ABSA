import torch

label_map = {  #因为论文中的输出为 POS 等，为了更好的得到 label 的 embedding -> 使用完整的词
            'POS': '<<positive>>',
            'NEG': '<<negative>>',
            'NEU': '<<neutral>>'
        }
label2id_map = {   # 给 label id，用于最后的预测，此 id 也对应最后预测用的分布向量中的下标
            'POS': 1,
            'NEG': 2,
            'NEU': 3
        }

def cmp_aspect(v1, v2):
    if v1[0]['from']==v2[0]['from']:
        return v1[1]['from'] - v2[1]['from']
    return v1[0]['from'] - v2[0]['from']

def padding(x, max_len, pad_value = 0):
    """
    得到一个 padding 后的 tensor
    :param x: 可以是 list，也可以是 tensor
    :param max_len:
    :param pad_value:
    :return:
    """
    if isinstance(x, list):
        x_len = len(x)
        temp = torch.tensor(x)
        x = torch.full([max_len], pad_value)
        x[:x_len] = temp
    return x