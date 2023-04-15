import scapy.all as scapy
import binascii
import tokenizers
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import json
import pandas as pd
import re
import random
from transformers import BertTokenizer

random.seed(40)


dataset_dir = "D:\\ty\\phd\\2022.5\\domain-BERT\\dataset\\"


corpora_list=['d_labels.txt','top-1m-2020.csv']
corpora_write_file="corpora.txt"
vocab_write_file = "domain_vocab_all.txt"

def get_corpora():
    corpora_write=''
    for corpora_file in corpora_list:
        if corpora_file=='d_labels.txt':
            with open(dataset_dir+"d_labels.txt", "r") as fin:
                for line in fin.readlines():
                    line = line.strip()
                    corpora_write += re.split(':', line)[0]+'\n'
        if corpora_file=='top-1m-2020.csv':
            alexa_info = pd.read_csv(dataset_dir+corpora_file, header=None)
            domains = list(alexa_info.iloc[:, 1])
            for domain in domains:
                corpora_write += domain+'\n'
    with open(dataset_dir + corpora_write_file, 'a') as f:
        f.write(corpora_write)


def build_vocab():
    bwpt = tokenizers.BertWordPieceTokenizer()
    filepath = "./dataset/corpora.txt"  # 语料文件
    # 训练分词器
    bwpt.train(
        files=[filepath],
        vocab_size=50000,  # 这里预设定的词语大小不是很重要
        min_frequency=1,
        limit_alphabet=1000
    )
    # 保存训练后的模型词表
    bwpt.save_model('./models/')
    # output： ['./pretrained_models/vocab.txt']

    # 加载刚刚训练的tokenizer
    tokenizer = BertTokenizer(vocab_file='./models/vocab.txt')

if __name__ == '__main__':
    # build vocab
    #get_corpora()
    build_vocab()
    

