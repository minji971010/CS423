from img_similarity import get_similarity_dict
from pagerank import get_ranks_by_dict
import os
import pickle

sample_cfg = [
    {
        "pickle_name": "dog_cat_sim_dict",
        "sample_dirs": [
            "./downloads/cat_sample/",
            "./downloads/dog_sample/",
            "./downloads/dog_cat_sample/"
        ]
    },
    {
        "pickle_name": "youtube_sim_dict",
        "sample_dirs": [
            "./downloads/youtube_sample/"
        ]
    }
]

pickle_dir = "pickles"

cfg_idx = 1
sample_dirs = sample_cfg[cfg_idx]["sample_dirs"]
sample_cnt = 50
pickle_file_path = "%s/%s-%d.pickle" % (pickle_dir, sample_cfg[cfg_idx]["pickle_name"], sample_cnt)

imgs = []
for sample_dir in sample_dirs:
    imgs.extend(map(lambda fname: sample_dir + fname, os.listdir(sample_dir)[:sample_cnt]))
print(imgs)

if os.path.exists(pickle_file_path):
    with open(pickle_file_path, "rb") as pickle_file:
        sim_dict = pickle.load(pickle_file)
else:
    sim_dict = get_similarity_dict(imgs)
    with open(pickle_file_path, "wb") as pickle_file:
        pickle.dump(sim_dict, pickle_file)

for i, j in sim_dict:
    print("%s / %s : %f" % (imgs[i], imgs[j], sim_dict[(i, j)]))

rank_list = get_ranks_by_dict(imgs, sim_dict, threshold=0.04, max_loop=1000)
_max_idx = 0
_max = rank_list[0]
for i in range(len(rank_list)):
    if _max < rank_list[i]:
        _max_idx = i
        _max = rank_list[i]

print(_max_idx, _max, imgs[_max_idx])
    
