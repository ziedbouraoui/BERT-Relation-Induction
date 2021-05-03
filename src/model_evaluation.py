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

from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics import precision_score, recall_score, average_precision_score, f1_score

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss


from convert_input_to_examples import* 
import convert_examples_to_features
from util import*  

from multiprocessing import Pool, cpu_count

from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_transformers.optimization import  AdamW, WarmupLinearSchedule


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
################

relation_name = sys.argv[1]
log_path = init_logging_path('bert_eval',relation_name)
logging.basicConfig(filename=log_path, 
                    level=10,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    )

logging.info("----------------start logging ----------------" + '\n') 
logging.info("Log file: " + log_path) 


dir_data = os.path.join(os.getcwd(),f"data/{relation_name}")
dir_cache = os.path.join(os.getcwd(),"bert/cache/")
dir_bert = os.path.join(os.getcwd(),f"bert/output/{relation_name}")
dir_report = os.path.join(os.getcwd(),f"bert/report/{relation_name}")

MAX_SEQ_LENGTH = 128
TRAIN_BATCH_SIZE = 24
EVAL_BATCH_SIZE = 8
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
bert_version = 'bert-base-cased'
num_labels = 2 

def positive(s, i, j, ii, jj):
    s = s.replace(" "+i+" ",' MASKS ')
    s = s.replace(" "+j+" ",' MASKT ')
    s = s.replace(' MASKS '," "+ii+" ")
    s = s.replace(' MASKT '," "+jj+" ")
    return s

def negative(s, i, j, ii,jj):
    s = s.replace(" "+i+" ",' MASKS ')
    s = s.replace(" "+j+" ",' MASKT ')
    s = s.replace(' MASKS '," "+jj+" ")
    s = s.replace(' MASKT '," "+ii+" ")
    return s

def prepare_test_data(relation,df_test_data,df_train_patterns):
    random.seed(10)        
    df_pos_dev =  pd.DataFrame(columns=['Source', 'Target', 'Relation','Pattern'])   
    df_neg_dev =  pd.DataFrame(columns=['Source', 'Target', 'Relation','Pattern'])
    data_test_groupped = df_test_data.groupby(['Source','Target'])   
    for index, row in df_train_patterns.iterrows() : 
        source = row['Source']
        target = row['Target']
        pattern = row['Pattern']
        for name, group in data_test_groupped: 
            source_ex = name[0]
            target_ex = name[1]
            pp = positive(pattern, source, target, source_ex, target_ex)
            nn =  negative(pattern, source, target,source_ex, target_ex)
            df_pos_dev = df_pos_dev.append({'Source': source_ex, 'Target': target_ex, 'Relation': relation, 'Pattern':pp}, ignore_index=True)
            df_neg_dev = df_neg_dev.append({'Source': target_ex, 'Target': source_ex, 'Relation': relation, 'Pattern':nn}, ignore_index=True)
    df_pos_dev.insert(0, 'label', 1)
    df_neg_dev.insert(0, 'label', 0)
    df_dev = pd.concat([df_pos_dev,df_neg_dev]).reset_index().drop(columns="index")
    df_dev.insert(0, 'ID', range(len(df_dev)))          
    return df_dev 

################################


def compute_metrics(preds,positive_preds):
    pr = precision_score(positive_preds, preds,pos_label=1)
    rec = recall_score(positive_preds, preds,pos_label=1)
    f1 = f1_score(positive_preds, preds,pos_label=1)
    return {
        "pr": pr ,
        "rec": rec,
        "f1": f1
    }
    
def prepare_dir_report(report_dir): 
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)


def eval_bert_model(relation_label): 
    #
    dir_model = os.path.join(dir_bert,f'{relation_label}/{top_k}') 
    tokenizer = BertTokenizer.from_pretrained(dir_model  + '/vocab.txt', do_lower_case=False)
    

    relation_dir = os.path.join(dir_data,f"splits/{relation_label}")
    
    df_data = pd.read_csv(os.path.join(relation_dir, 'dev.tsv'), sep='\t', 
            names=['ID','label','Source', 'Target', 'Relation','Pattern'])

    df_data_groupped = df_data.groupby(['Source','Target'])   
    
    all_preds = []
    positive_preds =[]
    for name, group in df_data_groupped: 
        ##################
        processor = BinaryClassificationProcessor()
        eval_examples = processor.get_dev_examples2(group)
        label_list = processor.get_labels() 
        print(group['label'].unique()[0] )
        positive_preds.append(group['label'].unique()[0])
        eval_examples_len = len(eval_examples)
        
        label_map = {label: i for i, label in enumerate(label_list)}
        eval_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) 
                                        for example in eval_examples]
        
        if __name__ ==  '__main__':
            print(f'Preparing to convert {eval_examples_len} examples..')
            print(f'Spawning {process_count} processes..')
            with Pool(process_count) as p:
                eval_features = list(tqdm_notebook(p.imap(convert_examples_to_features.convert_example_to_feature, eval_examples_for_processing), total=eval_examples_len))

        # with open(os.path.join(dir_data,f"splits/{relation_label}/eval_features.pkl"), "wb") as f:
        #     pickle.dump(eval_features, f)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)        
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)
        ############################
        #################
         # Load pre-trained model (weights)
        model = BertForSequenceClassification.from_pretrained(dir_model , cache_dir=dir_model, num_labels=2)
        model.to(device)
        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds  =[]    
        
        for input_ids, input_mask, segment_ids, label_ids in tqdm_notebook(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                outputs = model(input_ids, segment_ids, input_mask, labels=label_ids)
                loss, logits = outputs[0], outputs[1]
                tmp_eval_loss = loss

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
  
            #print(preds)
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)
            
        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        #print(preds)
        #print("***") 
        preds_max = np.amax(preds, axis=0)
        #print(preds_max)
        #print("***") 
        preds_max = np.argmax(preds_max, axis=0)
        #print(preds_max)
        all_preds.append(preds_max)

    print(all_preds)
    ##################
    report_dir = os.path.join(dir_report,f"{relation_label}/{top_k}")
    prepare_dir_report(report_dir)
    result = compute_metrics(all_preds,positive_preds)
    print(result)
    result['eval_loss'] = eval_loss
    output_eval_file = os.path.join(report_dir, 'eval_results.txt')
    with open(output_eval_file, "w") as writer:
         logger.info("***** Eval results *****")
         for key in (result.keys()):
             logger.info("  %s = %s", key, str(result[key]))
             writer.write("%s = %s\n" % (key, str(result[key])))


##########################################

df_relation = pd.read_csv(os.path.join(dir_data, 'data.txt'), sep=' ', 
        names=['Source','Target','Relation'])

if __name__ ==  '__main__':    
    logging.info('Prepare test Data...')
    for name, group in df_relation.groupby(df_relation.Relation) : 
        logging.info(name)    
        relation_label = format_filename(name)
        relation_dir = os.path.join(dir_data,f"splits/{relation_label}")
            
        if os.path.isfile( os.path.join(relation_dir, 'train_patterns.tsv') ):
            df_train_patterns = pd.read_csv(os.path.join(relation_dir, 'train_patterns.tsv'), sep='\t', 
                names=['Source','Target','Relation','Pattern','Score'])

            df_test_data = pd.read_csv(os.path.join(relation_dir, 'dev_context.tsv'), sep='\t', 
                names=['Source','Target','Relation','Context'])
            
            df_train_patterns = df_train_patterns.nlargest(top_k,['Score'])
            
            df_dev = prepare_test_data(name,df_test_data,df_train_patterns)
            
            if not os.path.exists(relation_dir):   
                os.makedirs(relation_dir)
            with open(os.path.join(relation_dir,'dev.tsv'), 'w'):
                df_dev.to_csv(os.path.join(relation_dir,'dev.tsv'), header=None, index=None, sep='\t', mode='a')

            eval_bert_model(relation_label)

    