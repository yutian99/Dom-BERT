# README



## Stage1 *Neighbor Sampling on HIN*

run rawdata_preprocess.py to construct heterogeneous graph and sample neighbors for each domain node.





## Stage2 *Fine-tuning*

run  fine-tuning.py to do fine-tuning.



## Stage3 *Malicious domain detection*

run score2label.py to get the final label for each unlabeled domain according to prediction results in stage2.