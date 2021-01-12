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
    
    """
    df = pd.read_csv('/dlabdata1/lugeon/websites_alexa_mostpop.gz', index_col=0)
    url_list = df.url.values
    
    embeddings, errors = embed(url_list, mode='content', aggregation='average', workers=24)
    
    df['embedding'] = embeddings
    df['error'] = errors
    print(df)
    
    df.to_csv('/dlabdata1/lugeon/websites_alexa_mostpop_finalemb.gz', compression='gzip')
    """

    
    
    
    basefile = '/dlabdata1/lugeon/dmozfinalset/dmoz_en_full_train'
    ext = '_html.json.gz'
    nb_samples = 94648
    
    df = embed_from_json(json_file=basefile + ext, 
                         nb_samples=nb_samples, 
                         mode='full', 
                         aggregation='concat', 
                         workers=24,
                         chunksize=480)
    
    df.to_csv(basefile + '_finalemb.gz', compression='gzip')
    
    print(df)
    
    


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
            job = pool.apply_async(worker, args=(url_list[i:j], aggregation, mode), callback=update_pbar)
            jobs += [job]
    
    finally:
        pool.close()
        pool.join()
        pbar.close()
    
    embeddings = []
    errors = []
    
    for job in jobs:
        _embeddings, _errors = job.get()
        embeddings += _embeddings
        errors += _errors
        
    return embeddings, errors


def embed_from_json(json_file, nb_samples, aggregation='average', mode='full', chunksize=100, workers=4):
    
    reader = pd.read_json(json_file, orient='records', lines=True, chunksize=chunksize)
    
    pool = mp.Pool(workers)
    
    pbar = tqdm(total = nb_samples)
    
    batch_size = math.ceil(chunksize/workers)
    
    def update_pbar(item):
        pbar.update(batch_size)
        
    df_main = pd.DataFrame([])
        
    for df_chunk in reader:
        
        nb_urls = df_chunk.shape[0]

        jobs = []
        
        for i in range(0, nb_urls, batch_size):
            j = min(i+batch_size, nb_urls)
            subdf = df_chunk.iloc[i:j]
            job = pool.apply_async(worker, args=(subdf['url'].values, 
                                                 subdf[['errcode', 'content', 'html']],
                                                 aggregation, 
                                                 mode), 
                                   callback=update_pbar)
            jobs += [job]
    
        embeddings = []
        errors = []

        for job in jobs:
            _embeddings, _errors = job.get()
            embeddings += _embeddings
            errors += _errors
            
        df_chunk['embedding'] = embeddings
        df_chunk['error'] = errors
        
        df_main = pd.concat((df_main, df_chunk[['url', 'embedding', 'error', 'cat0']]))
        
        
        
    pool.close()
    pool.join()
    pbar.close()
    
    return df_main
            
        

def is_valid(code, content):
    return code == 200 and content.startswith('text/html')
        
    
    
    
    
def worker(url_sublist, html_df=None, aggregation='average', mode='textual'):
    
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    xlmr = SentenceTransformer('xlm-r-distilroberta-base-paraphrase-v1', device=device)
    
    embeddings = []
    errors = []
    
    for u in range(len(url_sublist)):
        
        url = url_sublist[u]
        
        if html_df is None:
            
            try:
                r = requests.get(url, timeout=20)
            except RequestException as e:
                #print('Request error with {}: {}'.format(url, e.__class__.__name__))
                embeddings += [np.nan]
                errors += [e.__class__.__name__]
                continue

            status_code = r.status_code
            content_type = r.headers.get('content-type', '').strip()
            html = r.text
            
        else:
            row = html_df.iloc[u]
            status_code = row.errcode
            content_type = row.content
            html = row.html
            
        url_ok = status_code == 200 and content_type.startswith('text/html')  

        if not url_ok:
            #print('Not a valid website {}'.format(url))
            embeddings += [np.nan]
            errors += ['invalid website']
            continue
            
        soup = BeautifulSoup(html, 'lxml')
        
        if mode == 'url':
            url_features = embed_url(url, xlmr)
            
            aggres = 'u:'
            features = [url_features]
            
        if mode == 'description':
            desc_features = embed_description(soup, xlmr)
            
            aggres = 'd:'
            features = [desc_features]
            
        if mode == 'content':
            text_features = embed_text(soup, xlmr)
            
            aggres = 'c:'
            features = [text_features]
            
        if mode == 'textual':
            kw_features = embed_keywords(soup, xlmr)
            desc_features = embed_description(soup, xlmr)
            text_features = embed_text(soup, xlmr)
            
            aggres = 'cdk:'
            features = [text_features, desc_features, kw_features]
            
        if mode == 'full':
            url_features = embed_url(url, xlmr)
            text_features = embed_text(soup, xlmr)
            desc_features = embed_description(soup, xlmr)
            kw_features = embed_keywords(soup, xlmr)
            
            aggres = 'ucdk:'
            features = [url_features, text_features, desc_features, kw_features]


        embedding = np.array([])
        
        counter = 0
        
        for f in features:
            if f is None:
                
                if counter == 0:
                    aggres += '_' * len(features)
                    break
                
                aggres += '_'
                
                if aggregation == 'average':
                    continue
                if aggregation == 'concat':
                    replacement = embedding[:counter*768].reshape(counter, 768).mean(axis=0)
                    embedding = np.concatenate((embedding, replacement), axis=0)
                    
            else:
                embedding = np.concatenate((embedding, f), axis=0)
                aggres += 'o'
                counter += 1
        
        if embedding.shape[0] == 0:
            embeddings += [np.nan]
            errors += [aggres]
            continue
            
        if aggregation == 'concat':
            embeddings += [embedding.tolist()]
            errors += [aggres]
                
        if aggregation == 'average':
            embeddings += [embedding.reshape(-1, 768).mean(axis=0).tolist()]
            errors += [aggres]
        
    return embeddings, errors
        
    
def embed_text(soup, transformer):
    
    sentences = split_in_sentences(soup)
    
    if len(sentences) == 0:
        return None
    
    text_emb = transformer.encode(sentences)
    
    if text_emb.size == 0:
        return None
    
    return text_emb.mean(axis=0) # mean of the sentences  

def embed_description(soup, transformer):
    
    desc = soup.find('meta', attrs = {'name': ['description', 'Description']})
    
    if not desc:
        return None
    
    content = desc.get('content', '')
    
    if len(content.strip()) == 0:
        return None
    
    desc_split = [s.strip() for s in content.split('.') if s]
    desc_emb = transformer.encode(desc_split)
    
    if desc_emb.size == 0:
        return None
    
    return desc_emb.mean(axis=0) # mean of the sentences  


def embed_keywords(soup, transformer):
    
    kw = soup.find('meta', attrs = {'name': 'keywords'})
    
    if not kw: 
        return None
    
    content = kw.get('content', '')

    if len(content.strip()) == 0:
        return None
    
    kw_emb = transformer.encode(content)
    
    if kw_emb.size == 0:
        return None
    
    return kw_emb 


def embed_graphical(url):
    return 

def embed_url(url, transformer):
    cleaned_url = clean_url(url)
    
    url_emb = transformer.encode(cleaned_url)
    
    if url_emb.size == 0:
        return None
    
    return url_emb.mean(axis=0)


def split_in_sentences(soup):
    """ From the raw html content of a website, extract the text visible to the user and splits it in sentences """
    
    sep = soup.get_text('[SEP]').split('[SEP]') # separate text elements with special separators [SEP]
    strip = [s.strip() for s in sep if s != '\n']
    clean = [s for s in strip if len(s) != 0]
    
    return clean


def clean_url(url):
    url = re.sub(r"www.|http://|https://|-|_", '', url)
    return url.split('.')[:-1]




if __name__ == '__main__':
    main()