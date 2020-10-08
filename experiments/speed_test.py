import pickle
import os
from tqdm.auto import tqdm

import timeit
import requests
import numpy as np
import json

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.fastest = False

# torch.set_num_threads(1)

base_url="http://18.203.115.216:5007/api/v1/reranking"

gpu_times_dict = {}
for x in sorted(os.listdir("../Search_data/")):
    title = "../Search_data/"+x
    if title.endswith("checkpoints"):
        continue

    batch_size = int(x.split("_")[2])
    
    if batch_size != 50:
        continue

    print("batch size : ", batch_size)


    gpu_times = []

    with open(title,'rb') as f:
        rerank_test = pickle.load(f)

    idx = 0
    for x in rerank_test:
        idx += 1
        query_string = x[0]
        master_question = x[1]
        hits = x[2]
        

        texts = [[0,x[1]] for x in hits]


        start = timeit.default_timer()
        params = {
            "query": query_string,
            "texts": texts,
        }
        r = requests.get(base_url, json=json.dumps(params))
        response  = r.json()
        scoreDocs = response['scoreDocs']

        stop = timeit.default_timer()
        gpu_times.append(stop-start)

        if idx%10==0:
            print("mean time : ", np.mean(gpu_times))

        if idx%30== 0:
            print("mean time : ", np.mean(gpu_times[-30:]))
        
        if idx%100==0:
            break