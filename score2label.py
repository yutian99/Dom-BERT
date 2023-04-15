import pandas as pd
import random
mal_set=set()
ben_set=set()

TP=0
FP=0
FN=0
TN=0
def get_label(df):
    global TP
    global FP
    global FN
    global TN
    ans=pd.DataFrame()
    label=0
    predict=0
    domain=df.name
    if domain in mal_set:
        label=1
    mal_df=df[df.sentence2.isin(mal_seed_set)]
    mal_df=mal_df[mal_df.cosine_scores>0.5]
    if len(mal_df)>0:
        predict=1
    if label==1 and predict==1:
        TP+=1
    elif label==0 and predict==0:
        TN+=1
    elif label==1 and predict==0:
        FN+=1
    else:
        FP+=1
    
    #if label!=predict:
        #print(df.name,label,predict)
    #print(mal_df[["cosine_scores","label"]])
    #print(df[["cosine_scores","label"]])
def evaluation():
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    F1=2*precision*recall/(precision+recall)
    print("TP:{}".format(TP))
    print("FP:{}".format(FP))
    print("FN:{}".format(FN))
    print("TN:{}".format(TN))
    print("precision:{}".format(precision))
    print("recall:{}".format(recall))
    print("accuracy:{}".format(accuracy))
    print("F1 score:{}".format(F1))
with open("./labeled_data.txt","r") as fin:
    for line in fin.readlines():
        line = line.strip()
        l = int(line[-1])
        if l == 1:
            mal_set.add(line[:-2])
        elif l==0:
            ben_set.add(line[:-2])
mal_seed_set=set()
while len(mal_seed_set)<100:
    domain=random.choice(list(mal_set))
    mal_seed_set.add(domain)
score_df=pd.read_csv("./similarity_evaluation_test_results.csv")
test_df=score_df[~score_df.sentence1.isin(mal_seed_set)]
test_df.groupby("sentence1").apply(get_label)
evaluation()
