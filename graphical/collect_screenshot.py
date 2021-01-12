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
    """ This script take screenshots and store them as jpeg files from a list of urls"""
    
    # file that contains the urls
    folder = "/dlabdata1/lugeon/dmozfinalset/"
    file_name = "dmoz_jap"
    ext = ".gz"
    
    test_train = False # if the files corresponds to already splitted samples, i.e end with "train" or "valid"
    
    if test_train:
        out_folder = folder + file_name[:-5] + "screenshots/" + file_name[-5:] + '/'
    else:
        out_folder = folder + file_name + "_screenshots/"
        
    # where the screenshots will be saved
    
    print('Out folder : ' + out_folder)
    
    print('Deleting all previous files')
    
    # remove all files in out folder
    dirs = glob.glob(out_folder + '*')
    for d in dirs:
        shutil.rmtree(d)
        
    print('Collecting screenshots from ' + file_name + '...')

    # reading dataframe
    df = pd.read_csv(folder + file_name + ext, header=0, names=['uid', 'url', 'cat0'])
    #df = df.sample(frac=1, random_state=42)
    #df = df.iloc[:100]
    
    for cat in df.cat0.unique():
        os.mkdir(out_folder + cat)
    

    # progess bar
    pbar = tqdm(total = df.shape[0])

    def update_pbar(item):
        pbar.update(chunk_size)
        
    chunk_size = 50
    
    pool = mp.Pool()
    
    manager = mp.Manager()
    
    q_success = manager.Queue()
    q_errors = manager.Queue()
    
    jobs = []

    # distribute the works within the workers
    for i in range(0, df.shape[0], chunk_size):
        job = pool.apply_async(worker, args=(i, i+chunk_size, df, out_folder, q_success, q_errors), callback=update_pbar)
        jobs += [job]
        
    for job in jobs:
        job.get()
        
    pool.close()
    pool.join()
    pbar.close()
    
    nb_success = q_success.qsize()
    nb_errors = q_errors.qsize()
    
    print('\n' + 'Success: {}, errors: {}'.format(nb_success, nb_errors) + '\n')
    
    errors = []
    
    while(q_errors.qsize()):
        errors += [q_errors.get()]
        
    print(pd.Series(errors).value_counts())
        
        



def take_screenshot(url, out_path, qs, qe):
    """ take a screenshot of a website and save it under the outpath """

    # dimensions of the browser window x1.3
    in_width = 1920 # 2496
    in_height = 1080 # 1404

    # dimension reduction, after the screenshot has been taken
    down_factor = 3
    out_width = int(in_width / down_factor)
    out_height = int(in_height / down_factor)

    # jpeg quality
    quality = 85

    
    try:
            
        timeout = 10
        
        r = requests.head('http://' + url, timeout=timeout)
        
        
        # check that the html status code is valid and that the accessed resource is an html page 
        # (and not a file that will be downloaded)
        status_code = str(r.status_code)
        content_type = r.headers.get('content-type', '').strip()
        
        two_status_code = status_code.startswith('2')
        three_status_code = status_code.startswith('3')
        
        nice_content_type = content_type.startswith('text/html')
        
        # because we used the head method rather than the get method, returned headers are different (e.g https websites)
        ok = (two_status_code and nice_content_type) or three_status_code
        
        if not(ok):
            qe.put((status_code, content_type))
            return
        
        # driver
        options = webdriver.ChromeOptions()
        options.headless = True
        
        # not download files automatically
        prefs = {}
        prefs["download.prompt_for_download"] = True
        prefs["download.default_directory"] = "/dlabdata1/lugeon/downloads"
        
        # disable windows like cookies -- doesn'r really work
        prefs["profile.default_content_setting_values.notifications"] = 2
        prefs["profile.default_content_settings.cookies"] = 2
        
        options.add_experimental_option("prefs", prefs)
        driver = webdriver.Chrome('/home/lugeon/drivers/chromedriver', options=options)
        
        driver.set_page_load_timeout(timeout)
        driver.set_window_size(in_width, in_height)

        # access the url, takes a screenshot (only in png)
        driver.get('http://' + url)
        time.sleep(2) # so that the website's elements are loaded
        driver.save_screenshot(out_path + '.png')

        # convert the png into a jpeg of lesser dimensions and quality
        img = Image.open(out_path + '.png')
        img = img.convert('RGB')
        img = img.resize((out_width, out_height), Image.ANTIALIAS)
        img.save(out_path + '.jpeg', optimize=True, quality=quality)
        os.remove(out_path + '.png')
        qs.put('ok')
        
    except RequestException as e:
        #print(e)
        qe.put(e.__class__.__name__)
        return
    except Exception as e:
        #print(e)
        qe.put(e.__class__.__name__)
        return

    finally:
        if 'driver' in locals():
            driver.quit()

def worker(start_id, end_id, df, out_folder, qs, qe):
    """ takes screenshots of websites that are within a slice of the dataframe """

    # in case the df length is not divisible by the step size
    if end_id > df.shape[0]:
        end_id = df.shape[0]

    sliced_df = df.iloc[start_id: end_id]
    
    sliced_df.apply(lambda row: take_screenshot(row.url, out_folder + str(row.cat0) + '/' + str(row.uid), qs, qe), axis=1) 

if __name__ == '__main__':
    main()