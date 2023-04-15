import os
import csv
from transformers import  BertTokenizer,WEIGHTS_NAME,TrainingArguments

import tokenizers

from transformers import (
    BertModel,
    BertForMaskedLM,
    BertForSequenceClassification,
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    LineByLineTextDataset
)
## 加载tokenizer和模型

token_path='./models/vocab.txt'
tokenizer =  BertTokenizer.from_pretrained(token_path, do_lower_case=True)
config_kwargs = {
    "cache_dir": None,
    "revision": 'main',
    "use_auth_token": None,
#      "hidden_size": 512,
#     "num_attention_heads": 4,
    "hidden_dropout_prob": 0.2,
     #"vocab_size": 50000 # 自己设置词汇大小
}
# 将模型的配置参数载入
config = AutoConfig.from_pretrained('bert-base-cased', **config_kwargs)
# 载入预训练模型
'''
model = AutoModelForMaskedLM.from_pretrained(
            'bert-base-cased',
            from_tf=bool(".ckpt" in 'roberta-base'), # 支持tf的权重
            config=config,
            cache_dir=None,
            revision='main',
            use_auth_token=None,
        )
'''
model = BertModel.from_pretrained("bert-base-cased",config=config)
#tokenizer = BertTokenizer(vocab_file='./models/vocab.txt')
model.resize_token_embeddings(len(tokenizer))

# 通过LineByLineTextDataset接口 加载数据 #长度设置为128, # 这里file_path于本文第一部分的语料格式一致
train_dataset=LineByLineTextDataset(tokenizer=tokenizer,file_path='./dataset/corpora.txt',block_size=128)
# MLM模型的数据DataCollator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
# 训练参数
pretrain_batch_size=64
num_train_epochs=300
training_args = TrainingArguments(output_dir='./outputs/', overwrite_output_dir=True, num_train_epochs=num_train_epochs, learning_rate=6e-5,
                                  per_device_train_batch_size=pretrain_batch_size, save_total_limit=10)
# 通过Trainer接口训练模型
trainer = Trainer(
    model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset)

# 开始训练
print("start")
trainer.train()
trainer.save_model('./outputs/')
