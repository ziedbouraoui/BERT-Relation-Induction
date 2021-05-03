from __future__ import print_function
from __future__ import division

import logging
#
import sys
import os
import pickle
import random 
#
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook, trange
#
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss, Linear
#
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_transformers.optimization import  AdamW, WarmupLinearSchedule
from pytorch_transformers import cached_path
#
from multiprocessing import Pool, cpu_count
#
from convert_input_to_examples import* 
import convert_examples_to_features
from util import*  


#
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#####


relation_name = sys.argv[1]
log_path = init_logging_path('bert_tuning',relation_name)
logging.basicConfig(filename=log_path, 
                    level=10,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    )

logging.info("----------------start logging ----------------" + '\n') 
logging.info("Log file: " + log_path) 


# # Set up Pathes and Parameters 
dir_data = os.path.join(os.getcwd(),f"data/{relation_name}")
dir_cache = os.path.join(os.getcwd(),"bert/cache/")
dir_output = os.path.join(os.getcwd(),f"bert/output/{relation_name}")


# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
MAX_SEQ_LENGTH = 128
TRAIN_BATCH_SIZE = 24
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 1
RANDOM_SEED = 42
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
OUTPUT_MODE = 'classification'
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
num_labels = 2
num_warmup_steps = 100
num_total_steps = 1000
n_gpu = torch.cuda.device_count()
logging.info(f"n_gpu = {n_gpu }" )
top_k = int(sys.argv[2])
process_count = cpu_count() - 1


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

###########################################

def swap(sentence, source, target):
    sentence = sentence.replace(" "+source+" ",' MASKS ')
    sentence = sentence.replace(" "+target+" ",' MASKT ')
    sentence = sentence.replace(' MASKS '," "+target+" ")
    sentence = sentence.replace(' MASKT '," "+source+" ")
    return sentence

def swap2(sentence, source, target, sample_source,sample_target):
    sentence = sentence.replace(" "+source+" ",' MASKS ')
    sentence = sentence.replace(" "+target+" ",' MASKT ')
    sentence = sentence.replace(' MASKS '," "+sample_source+" ")
    sentence = sentence.replace(' MASKT '," "+sample_target+" ")
    return sentence



def prepare_train_data(relation, df_data,df_other_relation):
    random.seed(10)
    sources = list(df_data['Source'].unique())
    targets = list(df_data['Target'].unique())

    other_sources = list(df_other_relationt['Source'].unique())
    other_targets = list(df_other_relationt['Target'].unique())
    
    df_pos_train = df_data.drop(columns="Score")
    df_pos_train.insert(0, 'label', 1)
    
    df_neg_train =  pd.DataFrame(columns=['Source', 'Target', 'Relation','Pattern'])
    data_groupped = df_data.groupby(['Source','Target'])
    for name, group in data_groupped : 
        source = name[0]
        target = name[1]
        s_list = list(sources)
        s_list.remove(source)
        t_list = list(targets)
        t_list.remove(target)
        for index, row in group.iterrows():
            pattern_switch1 = swap(row['Pattern'], source, target)
            df_neg_train  = df_neg_train.append({'Source': target, 'Target': source, 'Relation': relation, 'Pattern':pattern_switch1}, ignore_index=True)
            if len(s_list) > 1 & len(t_list) > 1 :
                # pattern 2
                sample_source = sample(s_list,1)
                sample_target = sample(t_list,1)
                pattern_switch2 = swap2(row['Pattern'], source, target, sample_source, sample_target)
                df_neg_train  = df_neg_train.append({'Source': sample_source, 'Target': sample_target, 'Relation': relation, 'Pattern':pattern_switch2}, ignore_index=True)
                # pattern 3
                sample_other_source = sample(other_sources,1)
                sample_other_target = sample(other_targets,1)
                pattern_switch3 = swap2(row['Pattern'], source, target, sample_other_source, sample_other_target)
                df_neg_train  = df_neg_train.append({'Source': sample_other_source, 'Target': sample_other_target, 'Relation': relation, 'Pattern':pattern_switch3}, ignore_index=True)

    df_neg_train.insert(0, 'label', 0)
    df_train = pd.concat([df_pos_train,df_neg_train]).reset_index().drop(columns="index")
    df_train.insert(0, 'ID', range(len(df_train)))
    return df_train

##############################################

def training_bert():  
    bert_version = 'bert-base-cased'
    bert_model = BertForSequenceClassification.from_pretrained(bert_version, cache_dir=dir_cache, num_labels=num_labels)
    bert_tokenizer = BertTokenizer.from_pretrained(bert_version, do_lower_case=False)    
    bert_model.to(device)

    relation_dir = os.path.join(dir_data,f"splits/{relation_label}")
    #pickle_path = os.path.join(relation_dir,"train_features.pkl")
    output_dir = os.path.join(dir_output,f"{relation_label}/{top_k}")
    
    prepare_dir_output(output_dir)

    # if os.path.isfile(pickle_path):
    #     train_features,num_train_optimization_steps = pickle.load(open(pickle_path, "rb"))
    # else:
    processor = BinaryClassificationProcessor()
    train_examples = processor.get_train_examples(relation_dir)    
    train_examples_len = len(train_examples)
    label_list = processor.get_labels() # [0, 1] for binary classification
    num_train_optimization_steps = int(
    train_examples_len / TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS) * NUM_TRAIN_EPOCHS
    # Load pre-trained model tokenizer (vocabulary)
    label_map = {label: i for i, label in enumerate(label_list)}
    train_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, bert_tokenizer, OUTPUT_MODE) 
                                         for example in train_examples]
        
    if __name__ ==  '__main__':
        logging.info(f'Preparing to convert {train_examples_len} examples..')
        logging.info(f'Spawning {process_count} processes..')
        with Pool(process_count) as p:
                train_features = list(tqdm_notebook(p.imap(convert_examples_to_features.convert_example_to_feature, train_examples_for_processing), total=train_examples_len))
    with open(os.path.join(dir_data,f"splits/{relation_label}/train_features.pkl"), "wb") as f:
            pickle.dump([train_features,num_train_optimization_steps], f) 

    param_optimizer = list(bert_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_train_optimization_steps)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    
    if n_gpu > 1:
        bert_model = torch.nn.DataParallel(bert_model)

    logger.info("***** Running training *****")
    logger.info("  Batch size = %d", TRAIN_BATCH_SIZE)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    #if OUTPUT_MODE == "classification":
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    #elif OUTPUT_MODE == "regression":
        #all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)
    
    train_data= TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    logger.info("train data = %d", len(train_data))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE)
    bert_model.classifier = LogisticRegression(768,2)

    #loss_fct = CrossEntropyLoss()
    bert_model.train()
    for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm_notebook(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            outputs = bert_model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
            loss, logits = outputs[0], outputs[1]
            # print(logits.view(-1, num_labels))
            # print( label_ids.view(-1))
            #loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

            if n_gpu > 1:
                loss = loss.mean() 
            
            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            loss.backward()
            logger.info(f'loss = {loss}')

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                
    model_to_save = bert_model.module if hasattr(bert_model, 'module') else bert_model  
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join( output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(),output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    bert_tokenizer.save_vocabulary(output_dir)

def prepare_dir_output(dir_output): 
    # if os.path.exists(dir_output) and os.listdir(dir_output):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(dir_output))
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

if __name__ ==  '__main__':

    df_relation_patterns = pd.read_csv(os.path.join(dir_data, 'relation_patterns.tsv'), sep='\t', 
        names=['Source','Target','Relation','Pattern','Score'])

    df_relation = pd.read_csv(os.path.join(dir_data, 'pairs_train.txt'), sep=' ', 
            names=['Source','Target','Relation'])
    
    for name, group in df_relation_patterns.groupby(df_relation_patterns.Relation) : 
        logging.info(name)
        relation_label = format_filename(name)
        relation_dir = os.path.join(dir_data,f"splits/{relation_label}")
        df_data = group.nlargest(top_k,['Score'])
        df_other_relation = df_relation[df_relation.Relation != str(name)]
        df_train = prepare_train_data(name,df_data,df_other_relation)
        

        if not os.path.exists(relation_dir):   
                os.makedirs(relation_dir)
        with open(os.path.join(relation_dir,'train.tsv'), 'w'):
                df_train.to_csv(os.path.join(relation_dir,'train.tsv'), header=None, index=None, sep='\t', mode='a')
           
        training_bert()

