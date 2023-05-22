
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np

import random

TRAIN=False
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout





#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else 'distilbert-base-uncased'

# Read the dataset
train_batch_size = 16
num_epochs = 1
#model_save_path = 'output/training_domain_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
data_path="data_all_walk_10"
model_save_path = 'output/training_domain_'+model_name.replace("/", "-")+'-'+data_path

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Convert the dataset to a DataLoader ready for training
logging.info("Read train dataset")

samples=[]
train_samples = []
dev_samples = []
test_samples = []

with open("./dataset/"+data_path+".txt", "r") as fin:
        for line in fin.readlines():
            line = line.strip()
            s1,s2,label=line.split(" ")[0],line.split(" ")[1],line.split(" ")[2]
            inp_example = InputExample(texts=[s1,s2], label=float(label))
            samples.append(inp_example)

random.shuffle(samples)
sample_size=len(samples)
logging.info("Size of dataset:{}\n".format(sample_size))
train_samples=samples[:sample_size//5*4][:1024] #先只取1024个sample进行代码测试
dev_samples=samples[sample_size//5*4:sample_size//10*9]
test_samples=samples[sample_size//10*9:]

if TRAIN:
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)


    logging.info("Read dev dataset")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


    # Configure the training. We skip evaluation in this example
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))


    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=model_save_path)


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################
logging.info("load model...")
model = SentenceTransformer(model_save_path)
#test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
#test_evaluator(model, output_path=model_save_path)
name="test"
csv_file = "similarity_evaluation"+("_"+name if name else '')+"_results.csv"
csv_headers = ["sentence1","sentence2","cosine_scores","manhattan_distances","euclidean_distances","dot_products","label"]

sentences1 = []
sentences2 = []
scores = []

for example in test_samples:
    sentences1.append(example.texts[0])
    sentences2.append(example.texts[1])
    scores.append(example.label)
embeddings1 = model.encode(sentences1, batch_size=16, show_progress_bar=False, convert_to_numpy=True)
embeddings2 = model.encode(sentences2, batch_size=16, show_progress_bar=False, convert_to_numpy=True)
labels = scores
cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

logging.info(embeddings1.shape)
logging.info(len(cosine_scores))
logging.info(len(manhattan_distances))
logging.info(len(dot_products))

eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

eval_pearson_dot, _ = pearsonr(labels, dot_products)
eval_spearman_dot, _ = spearmanr(labels, dot_products)
logging.info("Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_cosine, eval_spearman_cosine))
logging.info("Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_manhattan, eval_spearman_manhattan))
logging.info("Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_euclidean, eval_spearman_euclidean))
logging.info("Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_dot, eval_spearman_dot))
csv_path = os.path.join(model_save_path, csv_file)
output_file_exists = os.path.isfile(csv_path)
with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
    writer = csv.writer(f)
    if not output_file_exists:
        writer.writerow(csv_headers)
    for i in range(len(sentences1)):
        writer.writerow([sentences1[i],sentences2[i],cosine_scores[i],manhattan_distances[i],euclidean_distances[i],dot_products[i],labels[i]])


logging.info("test finished.")
