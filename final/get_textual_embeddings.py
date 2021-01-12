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
    This script compute the textual embeddings from a json file containing HTML responses,
    The embedding are then stored into a dataframe and written on the disk
    """
    

    basefile = '/dlabdata1/lugeon/dmozfinalset/dmoz_jap'
    ext = '_html.json.gz'
    nb_samples = 10400
    
    df = embed_from_json(json_file=basefile + ext, 
                         nb_samples=nb_samples, 
                         mode='textual', 
                         workers=24,
                         chunksize=300)
    
    df.to_csv(basefile + '_text_embeddings.gz', compression='gzip')
    
    print(df)
    


def embed_from_json(json_file, nb_samples, mode='full', chunksize=100, workers=4):
    
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
            job = pool.apply_async(worker, args=(subdf, mode), callback=update_pbar)
            jobs += [job]

        for job in jobs:
            df_chunk = job.get()
            df_main = pd.concat((df_main, df_chunk))
        
    pool.close()
    pool.join()
    pbar.close()
    
    return df_main
            
        

def is_valid(code, content):
    return code == 200 and content.startswith('text/html')
        

def get_feat_status(feature):
    return '_' if (feature is None) else 'o'
    
    
def worker(subdf, mode='textual'):
    
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    xlmr = SentenceTransformer('xlm-r-distilroberta-base-paraphrase-v1', device=device)
    
    features_fonctions = mode_dict[mode]
    
    # must manually put these
    embeddings = {'error': [], 
                  'embed_url': [], 
                  'embed_domain': [],
                  'embed_metatags': []}
    
    for f in features_fonctions:
        embeddings[f.__name__] = []        
    
    for website in subdf.itertuples():
        
        prefix = ''
        status = ''
        
        valid = is_valid(website.errcode, website.content) 

        if not valid:
            for key in embeddings.keys():
                embeddings[key].append(np.nan)
            continue
            
        # embed url
        _url_feature = embed_url(website.url, xlmr)
        url_feature =  np.nan if (_url_feature is None) else _url_feature.tolist()
        embeddings['embed_url'].append(url_feature)
        prefix += prefix_dict['embed_url']
        status += get_feat_status(_url_feature)
        
        # embed domain 
        domain_feature = embed_domain(website.url)
        embeddings['embed_domain'].append(domain_feature)
        prefix += prefix_dict['embed_domain']
        status += 'o' # we can always compute this
            
        soup = BeautifulSoup(website.html, 'lxml')
        
        # embed metatags
        metatags_features = embed_metatags(soup)
        embeddings['embed_metatags'].append(metatags_features)
        prefix += prefix_dict['embed_metatags']
        status += 'o' # we can always compute this

            
        # embed xlmr features 
        for function in features_fonctions:
            
            _feature = function(soup, xlmr)
            feature = np.nan if (_feature is None) else _feature.tolist()
            embeddings[function.__name__].append(feature)
            
            prefix += prefix_dict[function.__name__]
            status += get_feat_status(_feature)
            
        embeddings['error'].append(prefix + ':' + status)
                
    features_df = pd.DataFrame(embeddings)
    features_df.index = subdf.index

    return pd.concat((subdf[['uid', 'url', 'cat0']], features_df), axis=1)
                
        
    
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

def embed_title(soup, transformer):
    
    title = soup.find('title')
    
    if title is None:
        return None
    
    title = str(title.string)
    title = clean_title(title)
    
    if len(title) == 0:
        return None
    
    title_emb = transformer.encode(title)
    
    if title_emb.size == 0:
        return None
    
    return title_emb

def embed_links(soup, transformer):
    
    a_tags = soup.find_all('a', href=True)
    
    links = [a.get('href', '') for a in a_tags]
    links = [clean_link(link) for link in links]
    links = [link for link in links if len(link) != 0]
    
    words = [w for w in ' '.join(links).split(' ') if len(w) != 0]
    
    if len(words) == 0:
        return None
    
    n_words = 10
    most_frequent_words = pd.Series(words).value_counts()[:n_words].index.values
    
    if len(most_frequent_words) == 0:
        return None
    
    links_emb = transformer.encode(most_frequent_words)
    
    if links_emb.size == 0:
        return None
    
    return links_emb.mean(axis=0) # mean of the words 
    

def embed_graphical(url):
    return 

def embed_url(url, transformer):
    cleaned_url = clean_url(url)
    
    url_emb = transformer.encode(cleaned_url)
    
    if url_emb.size == 0:
        return None
    
    return url_emb.mean(axis=0)


def embed_domain(url):
    domain = url.split('.')[-1]
    return [float(domain.startswith(d)) for d in rep_domains]


def embed_metatags(soup):
    metatags = soup.findAll('meta')
    attr = [m.get('name', None) for m in metatags]
    attr = [a.lower() for a in attr if a != None]
    
    attr_emb = [float(a in attr) for a in repr_attributes]
    
    return attr_emb


def split_in_sentences(soup):
    """ From the raw html content of a website, extract the text visible to the user and splits it in sentences """
    
    sep = soup.get_text('[SEP]').split('[SEP]') # separate text elements with special separators [SEP]
    strip = [s.strip() for s in sep if s != '\n']
    clean = [s for s in strip if len(s) != 0]
    
    return clean

def clean_url(url):
    url = re.sub(r"www.|http://|https://|-|_", '', url)
    return url.split('.')[:-1]

def clean_title(title: str):
    title = re.sub(r"\*|[\n]|\||:", '', title)
    return title.strip()

def clean_link(link):
    link = re.sub(r"www.|http://|https://|[0-9]+", '', link)
    link = re.sub(r"-|_|=|\?|:", ' ', link)
    link = link.split('/')[1:]
    return ' '.join(link).strip()



mode_dict = {
    'textual': [embed_text, embed_title, embed_description, embed_keywords, embed_links]
}

prefix_dict = {
    'embed_url': 'u',
    'embed_text': 'c',
    'embed_title': 't',
    'embed_description': 'd',
    'embed_keywords': 'k',
    'embed_links': 'l',
    'embed_domain': 'i',
    'embed_metatags': 'm'
}


rep_domains = ['com', 'org', 'net', 'edu', 'info', 'biz', 'gov', 'tv', 'me', 'name', 'coop', 'pro', 'fm', 'int', 'aero', 'club', 'church']

repr_attributes = ['viewport','generator', 'description', 'twitter:site', 'twitter:card', 'robots', 'keywords', 'apple-mobile-web-app-capable', 'google-site-verification', 'msapplication-tileimage', 'msvalidate.01', 'apple-mobile-web-app-status-bar-style', 'shopify-digital-wallet', 'handheldfriendly', 'shopify-checkout-api-token', 'title', 'twitter:title', 'mobileoptimized', 'author', 'theme-color']




if __name__ == '__main__':
    main()