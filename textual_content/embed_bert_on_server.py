import pandas as pd
import numpy as np
import gensim.downloader as api
import re
from bs4 import BeautifulSoup
import multiprocessing as mp

from transformers import BertTokenizer, BertModel
import torch

import sys

from tqdm import tqdm

from bert_serving.client import BertClient

from helper import bert_split

folder = "/dlabdata1/lugeon/"
name = "websites_alexa_100_17cat"
ext = "_html.json.gz"
nb_samples = 1700

print('computing embeddings for ' + name + ext)

print('loading file...')

df = pd.read_json(folder + name + ext, orient='records', lines=True)
df = df[df.errcode == 200]
df = df[df.html.notnull()]

print('parsing html bodies...')

nb_cpu = 8
pool = mp.Pool(nb_cpu)

df['parsed_html'] = pool.map(bert_split, df.html)

pool.close()
pool.join()

df = df[df.parsed_html.notnull()]

non_empty = df.parsed_html.apply(lambda x: len(x) != 0)
df = df[non_empty]

bc = BertClient(ip='iccluster037.iccluster.epfl.ch', check_length=False)
            
print('connection with server established')

step = 100

embeddings = pd.Series(dtype='float64')

pbar = tqdm(total = df.shape[0])

for i in range(0, df.shape[0], step):
    
    j = i + step
    if j > df.shape[0]:
        j = df.shape[0]
        
    chunk_emb = df.iloc[i:j].apply(lambda row: bc.encode(row.parsed_html).mean(axis=0).tolist(), axis=1)
    
    embeddings = embeddings.append(chunk_emb)
    
    pbar.update(j-i)
    
pbar.close()
    
df['emb'] = embeddings

df = df[['uid', 'emb', 'cat']]

out_path = folder + name + "_emb_" + 'bert' + ".gz"

print('writing dataframe in ' + out_path)

df.reset_index(drop=True, inplace=True)

df.to_csv(out_path, compression='gzip')




    

        
    
