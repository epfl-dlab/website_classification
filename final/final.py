import requests
import re
import math
from tqdm import tqdm


import pandas as pd
import numpy as np

import multiprocessing as mp

from bs4 import BeautifulSoup

from sentence_transformers import SentenceTransformer

import torch

from requests.exceptions import RequestException


def main():
    
    df = pd.read_csv('/dlabdata1/lugeon/websites_alexa_mostpop.gz', index_col=0)
    url_list = df.url.values
    
    embeddings = embed(url_list, workers=24)
    
    df['embeddings'] = embeddings
    print(df)
    
    df.to_csv('/dlabdata1/lugeon/websites_alexa_mostpop_finalemb.gz', compression='gzip')


def embed(url_list, aggregation='average', mode='textual', workers=4):

    pool = mp.Pool(workers)
    
    nb_urls = len(url_list)
    
    batch_size = math.ceil(nb_urls/workers)
    
    jobs = []
    
    print('computing embeddings...')
    
    pbar = tqdm(total = nb_urls)

    def update_pbar(item):
        pbar.update(batch_size)
    
    try:
        for i in range(0, nb_urls, batch_size):
            j = min(i+batch_size, nb_urls)
            job = pool.apply_async(worker, args=(url_list[i:j],), callback=update_pbar)
            jobs += [job]
    
    finally:
        pool.close()
        pool.join()
    
    results = []
    
    for job in jobs:
        results += job.get()
        
    return results
        
def worker(url_sublist, aggregation='average', mode='textual'):
    
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    xlmr = SentenceTransformer('xlm-r-distilroberta-base-paraphrase-v1', device=device)
    
    embeddings = []
    
    for u in range(len(url_sublist)):
        
        url = url_sublist[u]
        try:
            r = requests.get(url, timeout=20)
        except RequestException as e:
            print('Request error with {}: {}'.format(url, e.__class__.__name__))
            embeddings += ['error']
            continue
        
        status_code = r.status_code
        content_type = r.headers.get('content-type', '').strip()
        
        url_ok = status_code == 200 and content_type.startswith('text/html')  
        
        if not url_ok:
            print('Not a valid homepage {}'.format(url))
            embeddings += ['error']
            continue
        
        soup = BeautifulSoup(r.text, 'lxml')
        
        url_features = embed_url(url, xlmr)
        desc_features = embed_description(soup, xlmr)
        text_features = embed_text(soup, xlmr)
        
        features = [url_features, desc_features, text_features]
        
        embedding = np.array([])
        
        for f in features:
            if f is None:
                continue
            else:
                embedding = np.concatenate((embedding, f))
        
        if embedding.shape[0] == 0:
            embeddings += ['error']
                
        if aggregation == 'average':
            embeddings += [embedding.reshape(-1, 768).mean(axis=0).tolist()]
        
    return embeddings
        

def embed_from_json(json_file, aggregation='average', mode='full'):
    return

def embed_text(soup, transfomer):
    
    sentences = split_in_sentences(soup)
    
    if len(sentences) == 0:
        return None
    
    features = transfomer.encode(sentences).mean(axis=0) # mean of the sentences  
    
    return features

def embed_description(soup, transfomer):
    
    desc = soup.find('meta', attrs = {'name': ['description', 'Description']})
    
    if not desc or len(desc) == 0:
        return None
    
    desc_split = [s.strip() for s in desc["content"].split('.') if s]
    features = transfomer.encode(desc_split).mean(axis=0) # mean of the sentences  
    
    return features


def embed_graphical(url):
    return 

def embed_url(url, transformer):
    cleaned_url = clean_url(url)
    return transformer.encode(cleaned_url)

def split_in_sentences(soup):
    """ From the raw html content of a website, extract the text visible to the user and splits it in sentences """
    
    sep = soup.get_text('[SEP]').split('[SEP]') # separate text elements with special separators [SEP]
    split = [s.split('.') for s in sep if len(s) != 0] # split text elements in sentences
    flat = [s for sublist in split for s in sublist] # flatten everything
    
    return flat


def clean_url(url):
    url = re.sub(r"www.|http://|https://|-|_", '', url)
    return url.split('.')[0]




if __name__ == '__main__':
    main()