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


def main():
    """ Entry point """
    
    folder = "/dlabdata1/lugeon/"
    name = "websites_alexa_10000_5cat"
    ext = "_html.json.gz"
    nb_samples = 50000

    print('computing embeddings for ' + name + ext)
    
    # reading file by chunks, so that each chunk can fit in the memory
    # the chunks should be as large as possible fo best parallelism performances
    chunksize = 10
    reader = pd.read_json(folder + name + ext, orient='records', lines=True, chunksize=chunksize) #, nrows=100) 

    df_main = pd.DataFrame([])
    #nb_cpu = int(mp.cpu_count() / 4)
    nb_cpu = 1
    pool = mp.Pool(nb_cpu)
    
    pbar = tqdm(total = nb_samples)
    
    # embedding function

    for chunk in reader:
        df_chunk = pd.DataFrame(chunk)
        df_chunk = clean_df(df_chunk)
        df_chunk = create_embeddings(df_chunk, embed_html, pool) # distribute the embedding computation over the workers
        df_main = pd.concat((df_main, df_chunk))
        pbar.update(chunksize)
        
    pool.close()
    pool.join()
    pbar.close()

    out_path = folder + name + "_emb_" + model_abbr + ".gz"
    print('writing dataframe in ' + out_path)
    
    df_main.reset_index(drop=True, inplace=True)
    
    df_main.to_csv(out_path, compression='gzip')
    
def clean_df(df):
    """ Remove all none elements of a dataframe """
    
    df_valid = df[df.errcode == 200]
    df_valid = df_valid[df_valid.html.notnull()]
    
    return df_valid

def create_embeddings(df, f, pool):
    """ Compute and returns the embeddings of html bodies using multiprocessing """
    
    print('chunk distributed')
    
    df['emb'] = pool.map(f, df.html) 
    return df[['uid', 'emb', 'cat']]




# ************************************ W2V ************************************

def clean_html(body):
    """ Parse the html body and returns only the words appearing on the website """
    
    try:
        soup = BeautifulSoup(body, 'html.parser')
        a = soup.get_text(' ').replace('\n', '').lower()
        b = re.sub(r"[^a-z ]+","", a)
        c = ' '.join(b.split())
        return c
    except:
        return ''

def clean_and_embed(body):
    """ Clean and compute the embedding of an html body """
    body_clean = clean_html(body)
    return word2vec_avg(body_clean)


def word2vec_avg(sentence):
    """ Compute the words embeddings in a sentence and average them """
    
    acc = np.zeros(300)
    counter = 0
    for word in sentence.split(' '):
        try:
            vector = model[word]
            acc += vector
            counter += 1
        except:
            pass
    if counter != 0:
        return (acc / counter).tolist()
    else:
        return None

    
# ************************************ BERT ************************************


def bert_clean_sentence(s):
    """ Clean a sentence so that it is only composed of words """
    
    s = re.sub(r"’", "'", s) # transform the french ' into english ones
    
    match = r"[^a-zA-z\u00C0-\u00FF '’-]" # match anything that is not a letter (incl. accentued), space, apostrophe and dash 
    match += "|" + "(\W\-|\-\W)+" # match any dash that is not between letters
    match += "|" + "(\W'|'\W)+" # match any apostrophe that is not between letters
    
    s = re.sub(match, "", s) # remove the matches characters
    s = re.sub(r"\s+"," ", s) # replace any whitespace with a space
    s = re.sub(r" +"," ", s) # remove any sucession of spaces
    s = s.strip() # trim the final sentence
    
    return s


def bert_split(body):
    """ From the raw html content of a website, extract the text visible to the user and splits it in sentences """
    
    soup = BeautifulSoup(body, 'html.parser')
    a = soup.get_text('[SEP]').split('[SEP]') # separate text elements with special separators, the, splits
    b = [s.split('.') for s in a if len(s) != 0] # split text elements in sentences
    flat_b = [bert_clean_sentence(s) for sublist in b for s in sublist] # clean the sentences, flatten everything
    cleaned_b = [s for s in flat_b if (len(s) != 0 and len(s) <=510)] # only keep sentences between 1 and 10 words
    
    return cleaned_b

def bert_embed_sentence(text):
    """ Get the BERT embedding of a sentence, aggregate by summing last to second layer of each word """
    
    tokenized_text = tokenizer.tokenize(text)
    
    # BERT model can only be fed with sentences shorter that 512 tokens
    if len(tokenized_text) > 510:
        tokenized_text = tokenized_text[:510]
        
    # special BERT tokens
    tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']
    
    # transforming tokens into ids, creating bert model input  
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens) # all the tokens belong to the first sentence
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    # to prevent torch for using several cores, parallelism is done at the main level
    torch.set_num_threads(1) 
    
    # passing through the BERT model
    with torch.no_grad():
        output = bert_model(tokens_tensor, segments_tensors)
        hidden_states = output[2] # tensor of shape 
    
    # Averaging words' last to second layer
    token_states = torch.stack(hidden_states, dim=0)
    token_states = torch.squeeze(token_states, dim=1) 
    token_embeddings = token_states[-2] 
    
    return torch.mean(token_embeddings, dim=0)

def bert_avg(body, server=False, bc=None):
    """ Get the embedding of an html body, that is the average of all the sentences """
    
    sentences = bert_split(body)
    
    print('html splitted')
    
    if server:
        return bc.encode(sentences).mean(axis=0)
    
    else:
        
        l = len(sentences)
    
        acc = np.zeros(768)

        for s in sentences:
            acc += bert_embed_sentence(s).numpy()

        if l != 0:
            return (acc / len(sentences)).tolist()
        else:
            None
        
    


# ************************************ ENTRY POINT ************************************

if __name__ == '__main__':
    
    args = sys.argv
    
    if args[1] == 'w2v':
        print('loading w2v model...')
        model = api.load('word2vec-google-news-300')
        model_abbr = 'w2v'
        
        embed_html = word2vec_avg
        
        main()
    
    elif args[1] == 'bert':
        
        
        if args[2] == 'server':
            
            bc = BertClient(ip='iccluster037.iccluster.epfl.ch', check_length=False)
            
            print('connection with server established')
            
            def bert_avg_server(body):
                return bert_avg(body, server=True, bc=bc)
            
            embed_html = bert_avg_server
        
        else:
                    
            print('loading bert model...')
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            bert_model = BertModel.from_pretrained('bert-base-cased', output_hidden_states = True)
            bert_model.eval()
            model_abbr = 'bert'

            embed_html = bert_avg

            
        
        main()
        
    else:
        print('error')
        
else:
    print('loading bert model...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bert_model = BertModel.from_pretrained('bert-base-cased', output_hidden_states = True)
    bert_model.eval()



    
