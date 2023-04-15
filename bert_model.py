from transformers import (
    CONFIG_MAPPING,MODEL_FOR_MASKED_LM_MAPPING, AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,DataCollatorForLanguageModeling,HfArgumentParser,Trainer,TrainingArguments,set_seed,
    BertTokenizer,
)
# 自己修改部分配置参数
config_kwargs = {
    "cache_dir": None,
    "revision": 'main',
    "use_auth_token": None,
#      "hidden_size": 512,
#     "num_attention_heads": 4,
    "hidden_dropout_prob": 0.2,
     "vocab_size": 50000 # 自己设置词汇大小
}
# 将模型的配置参数载入
config = AutoConfig.from_pretrained('bert-base-cased', **config_kwargs)
# 载入预训练模型
model = AutoModelForMaskedLM.from_pretrained(
            'bert-base-cased',
            from_tf=bool(".ckpt" in 'roberta-base'), # 支持tf的权重
            config=config,
            cache_dir=None,
            revision='main',
            use_auth_token=None,
        )
tokenizer = BertTokenizer(vocab_file='./models/vocab.txt')
model.resize_token_embeddings(len(tokenizer))
#output:Embedding(50000, 768, padding_idx=1)
