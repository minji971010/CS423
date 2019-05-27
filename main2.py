from img_similarity import get_similarity_dict
from pagerank import get_ranks_by_dict, isPairExist
import numpy as np
import os, pickle

sample_cfg = [
    {
        "pickle_name": "dog_cat_sim_dict",
        "sample_dirs": [
            "./samples/cat_sample/",
            "./samples/dog_sample/",
            "./samples/dog_cat_sample/"
        ]
    },
    {
        "pickle_name": "youtube_sim_dict",
        "sample_dirs": [
            "./samples/youtube_sample/"
        ]
    },
    {
        "sample_dirs": [
            "./samples/iron_sample/"
        ],
        "npy": "npy/iron.npy",
        "dim": 3,
        "img_cnt": 50
    },
    {
        "sample_dirs": [
            "./samples/new_dog_cat_sample/"
        ],
        "npy": "npy/dog_n_cat_asdf.npy",
        "img_cnt": 30
    },
    {
        "sample_dirs": [
            "./dog_n_cat/"
        ],
        "npy": "npy/30.npy",
        "dim": 1,
        "img_cnt": 30

    }
]

# USAGE: select sample to use
sample_idx = 4

cfg = sample_cfg[sample_idx]
sample_dirs = cfg["sample_dirs"]
sample_cnt = 30 if not "img_cnt" in cfg else cfg["img_cnt"]
prop_dim = 1 if not "dim" in cfg else cfg["dim"]
use_npy = "npy" in cfg

imgs = []
for sample_dir in sample_dirs:
    cnt = 0
    cursor = 0
    fnames = os.listdir(sample_dir)
    while cnt < sample_cnt:
        fname = sample_dir + fnames[cursor]
        if not (fname.endswith(".jpg") or fname.endswith(".jpeg")):
            cursor += 1
            continue
        imgs.append(fname)
        cnt += 1
        cursor += 1

print(imgs)

sim_dict_lst = [dict() for i in range(prop_dim)]

if use_npy:
    sim_list = np.load(cfg["npy"]).reshape((sample_cnt, sample_cnt - 1, prop_dim))
    # 0,0 -> 0,1
    # 0,1 -> 0,2
    # 1,0 -> 1,0
    # 1,1 -> 1,2
    for img_idx in range(len(sim_list)):
        for oidx in range(len(sim_list[img_idx])):
            real_oidx = oidx if img_idx > oidx else oidx + 1
            pair = (img_idx, real_oidx)
            if not isPairExist(pair, sim_dict_lst[0]):
                for val_idx in range(prop_dim):
                    sim_dict_lst[val_idx][pair] = sim_list[img_idx][oidx][val_idx]
else:
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, "rb") as pickle_file:
            sim_dict_lst[0] = pickle.load(pickle_file)
    else:
        sim_dict_lst[0] = get_similarity_dict(imgs)
        with open(pickle_file_path, "wb") as pickle_file:
            pickle.dump(sim_dict, pickle_file)





#print(sim_dict.keys())


rank_list_per_prop = []
for i in range(prop_dim):
    rank_list = get_ranks_by_dict(imgs, sim_dict_lst[i], threshold=0.8, max_loop=100)
    _max_idx = 0
    _max = rank_list[0]
    for i in range(len(rank_list)):
        if _max < rank_list[i]:
            _max_idx = i
            _max = rank_list[i]

    # print(_max_idx, _max, imgs[_max_idx])
    temp_list = list(reversed(sorted(rank_list)))[:10]

    for val in temp_list:
        print(val)
        for i in range(len(rank_list)):
            if(rank_list[i]==val):
                print(imgs[i])      
    rank_list_per_prop.append(rank_list)