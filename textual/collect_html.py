import numpy as np
import pandas as pd

import requests
import multiprocessing as mp
import json
import gzip

from tqdm import tqdm


def main():
    """"Entry point"""

    folder = '/dlabdata1/lugeon/dmozfinalset/'
    name = 'dmoz_jap'
    ext = '.gz'
    
    step = 100
    timeout = 10
    nb_samples = 10400

    data = pd.read_csv(folder + name + ext, header=0, index_col=0)
    data = data[['uid', 'url', 'cat0', 'lang']]
    
    print('retrieving html pages from', name)
    
    pbar = tqdm(total = nb_samples)
    
    def update_pbar(item):
        pbar.update(step)
        
    write_html_to_file(df=data, step=step, path=folder+name+'_html.json.gz', timeout=timeout, callback=update_pbar)


def get_homepage(url, timeout):
    """Return the html page corresponding to the url, or None if there was a request error"""

    try:
        r = requests.get("http://" + url, timeout=timeout)
        text = r.text
        code = r.status_code
        content = r.headers.get('content-type', '?').strip()
        return text, code, content
    except Exception as e:
        return None

def none_or_subscriptable(obj, pos):
    if obj != None:
        return obj[pos]
    else:
        return obj


def worker(start_id, end_id, df, q, timeout):
    """A worker that retrieve the html pages for a given slice of a dataframe and put them on the queue"""

    # in case the df length is not divisible by the step size
    if end_id > df.shape[0]:
        end_id = df.shape[0]

    # copying the df and retrieving the html pages
    sliced_df = df.iloc[start_id: end_id].copy()
    sliced_df['homepage'] = sliced_df.apply(lambda row: get_homepage(row.url, timeout), axis=1)

    # putting in the queue
    sliced_df.apply(lambda row: q.put({
        'uid': row.uid,
        'url': row.url,
        'html': none_or_subscriptable(row.homepage, 0),
        'errcode': none_or_subscriptable(row.homepage, 1),
        'content': none_or_subscriptable(row.homepage, 2),
        'cat0': row.cat0,
        'lang': row.lang
    }), axis=1)

    return



def listener(q, path, nb_lines):
    """A listener that takes elements in the queue and write them into a file, in json format"""

    i = 1 # to control the format

    with gzip.open(path, 'wt') as f:

        #f.write('[')
        f.flush()

        # listening on the queue
        while 1:
            m = q.get()

           # if all the workers are done
            if m == 'kill':
                #f.write(']')
                f.flush()
                print('done writing to file')
                break

            # else write the worker output on the file
            f.write(json.dumps(m))

            # for the format
            if(i < nb_lines):
                f.write('\n')
                i += 1

            f.flush()


def write_html_to_file(df, step, path, timeout, callback):
    """Retrieve and write in a file the html pages from urls in a dataframe"""

    manager = mp.Manager()
    q = manager.Queue()
    nb_cpu = int(mp.cpu_count() / 1) # can use all the cores, not really ressource-consuming 
    p = mp.Pool(nb_cpu)

    # put listener to work first, will occupy 1 thread
    watcher = p.apply_async(listener, (q, path, df.shape[0]))

    # fire off workers
    jobs = []

    # for each slice of the dataframe, a thread retreive the html pages
    for i in range(0, df.shape[0], step):
        job = p.apply_async(worker, (i, i+step, df, q, timeout), callback=callback)
        jobs.append(job)

    # to be sure that all jobs are done
    for job in jobs:
        job.get()

    # kill the listener
    q.put('kill')
    p.close()
    p.join()



if __name__ == "__main__":
    main()
