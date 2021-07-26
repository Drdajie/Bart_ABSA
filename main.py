import argparse
import random
import numpy as np
from pipe import BartABSAPipe
from bart_absa import BartSeq2SeqModel
from generator import SequenceGeneratorModel
from metrics import Seq2SeqSpanMetric
from fastNLP import BucketSampler, Trainer
from fastNLP.core.sampler import SortedSampler
import torch
from torch import optim
from my_loss import Seq2SeqLoss

def get_globel_varities(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model_name = "facebook/bart-base"
    args.model_path = "./model_parameters/model_state"
    args.bos_token_id, args.eos_token_id = 0, 1  # 自定义 mapping 中 <s> </s> 的位置
    args.max_len = 10   # 生成句子的最大长度, 每句话的decode长度为max_length + max_len_a*src_len
    max_len_a = {
        'penga/14lap': 0.9,
        'penga/14res': 1,
        'penga/15res': 1.2,
        'penga/16res': 0.9,
        'D20b/14lap': 1.1,
        'pengb/14res': 1.2,
        'pengb/15res': 0.9,
        'pengb/16res': 1.2
    }[args.dataset_name]
    args.max_len_a = max_len_a

def pre_prepare(args):
    # 设置种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def get_data(args):
    pipe = BartABSAPipe(tokenizer=args.model_name)#,opinion_first=args.opinion_first)
    data_bundle = pipe.process_from_file(args.data_dir)
    return data_bundle, pipe.tokenizer, pipe.mapping2id

def get_arameter_groups(args, model):
    parameters = []
    params = {'lr': args.lr, 'weight_decay': 1e-2}
    params['params'] = [param for name, param in model.named_parameters() if
                        not ('bart_encoder' in name or 'bart_decoder' in name)]
    parameters.append(params)

    params = {'lr': args.lr, 'weight_decay': 1e-2}
    params['params'] = []
    for name, param in model.named_parameters():
        if ('bart_encoder' in name or 'bart_decoder' in name) and not ('layernorm' in name or 'layer_norm' in name):
            params['params'].append(param)
    parameters.append(params)

    params = {'lr': args.lr, 'weight_decay': 0}
    params['params'] = []
    for name, param in model.named_parameters():
        if ('bart_encoder' in name or 'bart_decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
            params['params'].append(param)
    parameters.append(params)
    return parameters

def main():
    # 0_准备
    # 命令行解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default='AE', type=str,
                        choices=["AE", "OE"]
                        )
    parser.add_argument("--data_dir", default="./data/D20b/14lap/")
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--dataset_name', default='D20b/14lap', type=str)
    parser.add_argument("--output_dir", default="./results/")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epoch_size", default=50, type = int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--decoder_type', default=None, type=str, choices=['None', 'avg_score'])
    parser.add_argument('--use_encoder_mlp', type=int, default=1)
    parser.add_argument('--opinion_first', default=False)#, action='store_true')
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--num_beams', default=4, type=int)
    parser.add_argument('--length_penalty', default=1.0, type=float)
    args = parser.parse_args()

    # 预处理
    pre_prepare(args)
    get_globel_varities(args)
    # 准备数据
    data_bundle, tokenizer, mapping2id = get_data(args)
    args.label_ids = list(mapping2id.values())
    # 准备模型
    # 1_得到 Seq2SeqModel -> 用于训练得到模型参数，只能得到隐状态，不能得到生成的 seq
    model = BartSeq2SeqModel.build_model(args.model_name, tokenizer, label_ids=args.label_ids,
                                         decoder_type=args.decoder_type, copy_gate=False,
                                         use_encoder_mlp=args.use_encoder_mlp, use_recur_pos=False)
    # 2_得到 SeqGeneratorModel -> 想要 Seq2SeqModel 能生成 Seq 需要将其装进这个模型中。
    model = SequenceGeneratorModel(model, bos_token_id=args.bos_token_id, eos_token_id=args.eos_token_id,
                                   max_length=args.max_len, max_len_a=args.max_len_a, num_beams=args.num_beams,
                                   do_sample=False, repetition_penalty=1, length_penalty=args.length_penalty,
                                   pad_token_id=args.eos_token_id, restricter=None)

    # 训练阶段
    # 1_定义优化标准（损失函数）：定义在了 my_loss.py 中
    # 2_定义优化器
    param_groups = get_arameter_groups(args, model)
    optimizer = optim.AdamW(param_groups)
    # 3_定义衡量指标
    sampler = BucketSampler(seq_len_field_name='src_seq_len')
    metric = Seq2SeqSpanMetric(args.eos_token_id, num_labels=len(args.label_ids), opinion_first=args.opinion_first)
    # 4_gogogo
    model_path = None
    if args.save_model:
        model_path = 'save_models/'

    trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, optimizer=optimizer,
                      loss=Seq2SeqLoss(),
                      batch_size=args.batch_size, sampler=sampler, drop_last=False, update_every=1,
                      num_workers=2, n_epochs=args.epoch_size, print_every=1,
                      dev_data=data_bundle.get_dataset('dev'), metrics=metric, metric_key='triple_f',
                      validate_every=-1, save_path=model_path, use_tqdm=True, device=args.device,
                      # callbacks=callbacks,
                      check_code_level=0, test_use_tqdm=False)
                      #, test_sampler=SortedSampler('src_seq_len'), dev_batch_size=args.batch_size)

    trainer.train(load_best_model=False)

if __name__ == '__main__':
    main()