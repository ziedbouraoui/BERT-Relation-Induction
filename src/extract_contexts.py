import os
import logging
import sys, getopt


log_file = "log/" + sys.argv[1]
log_path = os.path.join(os.getcwd(),log_file)
if os.path.exists(log_path):
    os.remove(log_path )
with open(log_path , 'a'):
        os.utime(log_path , None)       

logging.basicConfig(filename=log_path, 
                    level=10,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    )
logging.info("----------------start logging ----------------" + '\n') 
logging.info("Log file: " + log_file) 


# -*- coding: utf-8 -*-
input_path_pairs= os.path.join(os.getcwd(), "data/" + sys.argv[2]) 
input_corpus_path= os.path.join(os.getcwd(), "data/corpus/" + sys.argv[3]) 
output_path= os.path.join(os.getcwd(), "data/" + sys.argv[4]) 

max_distance=15
max_sentence_length=100

def get_pairs(input_path_pairs):
    pair_file=open(input_path_pairs,encoding='utf-8').readlines()
    dict_pairs={}
    for line in pair_file:
        linesplit=line.strip().split("\t")
        bn1=linesplit[0]
        bn2=linesplit[1]
        if bn1 not in dict_pairs: dict_pairs[bn1]=set()
        #if bn2 not in dict_pairs: dict_pairs[bn2]=set()
        dict_pairs[bn1].add(bn2)
        #dict_pairs[bn2].add(bn1)
    return dict_pairs

logging.info("Starting...")                                        
#Load pairs from file
dict_pairs=get_pairs(input_path_pairs)
#Extract contexts
corpus_file=open(input_corpus_path,'r',encoding='utf-8')
dict_contexts={}
cont_lines=0
for line in corpus_file:
    cont_lines+=1
    #if cont_lines>1000: break
    if cont_lines%1000000==0: print (str(cont_lines/1000000)+"M lines processed")
    linesplit=line.strip().split(" ")
    if len(linesplit)>max_sentence_length: continue
    for i in range(len(linesplit)):
        token=linesplit[i]
        len_sentence=len(linesplit)
        if token in dict_pairs:
            for j in range(i+1,min(i+max_distance+1,len_sentence)):
                token_2=linesplit[j]
                if token_2 in dict_pairs[token]:
                    pair=token+"\t"+token_2
                    if pair not in dict_contexts: dict_contexts[pair]=set()
                    dict_contexts[pair].add(line)

logging.info("DONE GETTING CONTEXTS. Now printing on file...")                                        
txtfile=open(output_path,'w',encoding='utf-8')
set_pairs=set()
# for pair in dict_contexts:
#     txtfile.write(pair+"\n")
#     for sentence in dict_contexts[pair]:
#         txtfile.write(sentence)
#     txtfile.write("\n\n")
for pair in dict_contexts:
    for sentence in dict_contexts[pair]:
        txtfile.write(pair+"\t"+sentence)


txtfile.close()
logging.info("Finished")
