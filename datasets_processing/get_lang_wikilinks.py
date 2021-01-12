import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import multiprocessing as mp

import re

from langdetect import detect

from langdetect.detector_factory import init_factory

from langdetect.lang_detect_exception import LangDetectException

from requests.exceptions import RequestException

def init():
    init_factory()


def get_lang(url):
    try:
        r = requests.get('http://' + url, timeout=10)
        status_code = str(r.status_code)
        nice_status_code = status_code.startswith('2') or status_code.startswith('3')
        #nice_status_code = status_code in ['200', '203']
        nice_content_type = r.headers.get('content-type', '').startswith('text/html')
        
        if nice_status_code and nice_content_type:
            text = BeautifulSoup(r.text, 'lxml').get_text(' ')
            #text = re.sub(r"[^a-zA-z\u00C0-\u00FF '’-]", '', text)
            return detect(text)
        else:
            return None
    except RequestException as e:
        return None
    except Exception as e:
        print(e)
        return None

def foo(url):
    return None

def main():
    
    print('loading data')
    
    df = pd.read_csv("/dlabdata1/lugeon/wikilinks.gz", index_col=0)

    df_red = df.sample(20_000, random_state=42)

    #df_red = df
    
    pool = mp.Pool()

    print("Detecting languages")
    
    
    try:
        df_red['lang'] = pool.map(get_lang, df_red.link)
    finally:
        pool.close()
        pool.join()
    
        
    #df_red['lang'] = df_red.apply(lambda row: get_lang(row.link_url_normalised), axis=1)

    print('Writing to file')

    df_red.to_csv('/dlabdata1/lugeon/wikilinks_lang.gz', compression='gzip')

if __name__ == "__main__":
    main()