import logging
import sys
import os
import random 

#
import numpy as np
import pandas as pd
#
import torch
from pytorch_transformers import *
#
from multiprocessing import Pool, cpu_count
#
from util import *

#####

relation_name = sys.argv[1]

log_path = init_logging_path('pattern_extract',relation_name)

logging.basicConfig(filename=log_path, 
                    level=10,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    )

logging.info("----------------start logging ----------------" + '\n') 
logging.info("Log file: " + log_path) 


dir_data = os.path.join(os.getcwd(),f"data/{relation_name}")
dir_cache = os.path.join(os.getcwd(),"bert/cache/")

bert_model = BertForMaskedLM.from_pretrained("bert-large-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

process_count = cpu_count() - 1
top_k = 1000
len_context = 128
len_group = 5000
sep='\t'

##############################################################
##############################################################

def get_tokens(sentence):
    tokenized = bert_tokenizer.tokenize(sentence)
    tokenized = ['[CLS]'] + ['[MASK]' if x == 'mask' else x for x in tokenized] + ['[SEP]']
    mask_idx = [ idx for idx,x in enumerate(tokenized) if x == '[MASK]']
    token_ids = bert_tokenizer.convert_tokens_to_ids(tokenized)
    token_ids = torch.LongTensor(token_ids).unsqueeze(0)
    preds = bert_model(token_ids)[0]
    return preds[0,mask_idx]

def get_pattern_score(sentence,list_words):
        preds = get_tokens(sentence)
        probs_1,idx_1 = torch.topk(preds[0],10)
        preds_1 = bert_tokenizer.convert_ids_to_tokens(idx_1.numpy())
        intersection = [value for value in list_words if value in preds_1]
        return intersection 

def process_relation(group):
    if len(group) > len_group : 
        group = group.sample(n = len_group, random_state=100)
    records = []
    relation = group['Relation'].unique()[0]
    logging.info(relation+" : "+str(len(group)))
    sources = list(group['Source'].unique())
    targets = list(group['Target'].unique())
    for index, row in group.iterrows():
        source = " "+row['Source']+" " 
        target = " "+row['Target']+" "
        sentence = row['Context']
        source_sentence = sentence.replace(source, ' MASK ')
        target_sentence = sentence.replace(target, ' MASK ')
        if ("MASK" in source_sentence) & ("MASK" in target_sentence): 
            i_s = get_pattern_score(source_sentence,sources)
            i_t = get_pattern_score(target_sentence,targets)
            s = len(i_s)+len(i_t)
            if s > 0 :
                record = sep.join([row['Source'],row['Target'],relation,sentence,str(s)])
                records.append(record)          
    return records 


#########################################################
####################################################

def get_pattern_prediction(preds,word):
        predicted = 0.0
        probs_1,idx_1 = torch.topk(preds[0],10)
        preds_1 = bert_tokenizer.convert_ids_to_tokens(idx_1.numpy())
        if word in preds_1 : 
            predicted = 1.0
        return predicted 

def process_patterns(group):
    records = []
    name = group['Relation'].unique()
    relation = name[0]
    print(relation+" : "+str(len(group)))
    #sources = list(group['Source'].unique())
    #targets = list(group['Target'].unique())
    for index, row in group.iterrows():
        source = " "+row['Source']+" "
        target = " "+row['Target']+" "
        sentence = row['Pattern']
        source_sentence = sentence.replace(source, ' MASK ')
        target_sentence = sentence.replace(target, ' MASK ')
        if ("MASK" in source_sentence) & ("MASK" in target_sentence): 
            score_source = 0.0
            score_target = 0.0
            preds_s = get_tokens(source_sentence)
            preds_t = get_tokens(target_sentence)
            for index_1, row_1 in df_relation.loc[ (df_relation.Relation==relation) ].iterrows():
                score_source += get_pattern_prediction(preds_s,row_1['Source'])
                score_target += get_pattern_prediction(preds_t,row_1['Target'])
            score =  score_source + score_target 
            if score > 0 :
                record = '\t'.join([row['Source'],row['Target'],relation,sentence,str(score)])
                records.append(record)
    return records 


if __name__ ==  '__main__':

    df_relation_context = pd.read_csv(os.path.join(dir_data, 'relation_context.tsv'), sep='\t', 
        names=['Source','Target','Relation','Context'])
    df_relation_context['Context'] = df_relation_context['Context'].astype('str')
    mask = (df_relation_context['Context'].str.len() < len_context)
    df_relation_context = df_relation_context.loc[mask]


    logging.info('Preparing to convert context to pre patterns..')
    records = []
    with Pool(process_count) as p:
        tmp = p.map(process_relation, [group for name, group in df_relation_context.groupby(df_relation_context.Relation)])
        for item in tmp :
            for i in item :
                records.append(i)
    with open(os.path.join(dir_data, 'train_pre_patterns.tsv'), 'w') as f:
        for item in records:
             f.write("%s\n" % item)   
    logging.info('Done..')
    
    logging.info('Preparing to convert  pre-patterns to patterns ..')
    relation_pre_patterns = pd.read_csv(os.path.join(dir_data, 'train_pre_patterns.tsv'), sep='\t', 
        names=['Source','Target','Relation','Pattern', "Score"])

    df_relation = pd.read_csv(os.path.join(dir_data, 'pairs_train.txt'), sep=' ', 
            names=['Source','Target','Relation'])

    records_patterns = []   
    with Pool(process_count) as p:
            tmp = p.map(process_patterns, [group.nlargest(top_k,['Score']) for name, group in relation_pre_patterns.groupby(by="Relation")])
            for item in tmp :
                for i in item :
                  records_patterns.append(i)
     
    with open(os.path.join(dir_data, 'relation_patterns.tsv'), 'w') as f:
        for item in records_patterns:
             f.write("%s\n" % item)   
    logging.info('Done..')
            
