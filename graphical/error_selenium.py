import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests
from requests.exceptions import RequestException

from PIL import Image
import os
import glob
import shutil

import pandas as pd
import numpy as np

import multiprocessing as mp

from tqdm import tqdm

def main():
    """ entry point """
    
    # file that contains the urls
    folder = "/dlabdata1/lugeon/"
    file_name = "websites_wiki_10_000_9cat_train"
    ext = ".gz"

    # reading dataframe
    df = pd.read_csv(folder + file_name + ext, header=0, names=['uid', 'url', 'cat0'])
    
    df = df.sample(n=1000)

    def update_pbar(item):
        pbar.update(chunk_size)
        
    chunk_size = 10
    
    nb_cpu = int(48)
    #nb_cpu = 1
    pool = mp.Pool()
        
    df['error'] = pool.map(take_screenshot, df.url)

    pool.close()
    pool.join()

    df.to_csv(folder + 'selerrors.gz', compression='gzip')


def take_screenshot(url):
    """ take a screenshot of a website and save it under the outpath """

    # dimensions of the browser window
    in_width = 1920
    in_height = 1080

    # driver
    options = webdriver.ChromeOptions()
    options.headless = True
    
    # not download files automatically
    prefs = {}
    prefs["download.prompt_for_download"] = True
    prefs["download.default_directory"] = "/dlabdata1/lugeon/downloads"
    options.add_experimental_option("prefs", prefs)
    
    try:
    
        driver = webdriver.Chrome('/home/lugeon/drivers/chromedriver', options=options)
        
        timeout = 10

        driver.set_page_load_timeout(timeout)
        driver.set_window_size(in_width, in_height)
        
        r = requests.head('http://' + url, timeout=timeout)
        
        driver.get('http://' + url)
        driver.save_screenshot('/dlabdata1/lugeon/selerror.png')
        
        return 'ok'
        
    except Exception as e:
        #print(e)
        return e.__class__.__name__

    finally:
        if 'driver' in locals():
            driver.quit()

if __name__ == '__main__':
    main()
    