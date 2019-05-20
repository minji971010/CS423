# Rank(i) : 
#   (1 - df) / N(i) + df * sum_j(rank(j) / N(j)) 
#   where j is an image referencing to i (having similarity > threshold)
# N(i) : 
#   num of images which has similarity > threshold 
#   with image i (not including itself)

# -- INPUT --
# img_list = []   (currently not used) img1, img2, ...
# sim_dict = {}   (i,j) -> similarity of (img_i, img_j) (0 <= similarity <= 1)
#                 No (i,i) exists, Only one of (i, j) and (j, i) exists
# -- OUTPUT --
# rank_list = []  rank of each images
def get_ranks_by_dict(img_list, sim_dict, threshold=0.2, df=0.85, max_loop=100):
    img_num = len(img_list)

    # rank_list[i] : initial rank of image_i == 1/img_num
    # num_list[i]  : count of image referencing to (having high similarity with) image_i
    # ref_list[i]  : img indexes which have high similarity with img_i
    rank_list = [1.0 / img_num] * img_num
    num_list = [0] * img_num
    ref_list = [[] for i in range(img_num)] 
    
    # adjust similarity (> threshold -> 1 / o.w. -> 0) & set N(i)
    for pair in sim_dict:
        if sim_dict[pair] > threshold:
            i, j = pair
            num_list[i] += 1
            num_list[j] += 1
            sim_dict[pair] = 1
            ref_list[i].append(j)
            ref_list[j].append(i)
        else:
            sim_dict[pair] = 0
    print("num_list : ", num_list)
    # loop : update ranks
    cur_loop = 0
    while cur_loop < max_loop:
        # print ("loop %d : rank_list = " % cur_loop, rank_list)
        new_rank_list = []
        for i in range(img_num):
            if num_list[i] == 0:
                new_rank = rank_list[i]
            else:
                ref_rank_num_list = list(map(lambda idx: (rank_list[idx], num_list[idx]), ref_list[i]))
                norm_rank_sum = sum(map(lambda t: float(t[0]) / t[1], ref_rank_num_list)) # sum( Rank(other) / N(other) )
                new_rank = (1 - df) / num_list[i] + df * norm_rank_sum
            new_rank_list.append(new_rank)
        rank_list = new_rank_list
        cur_loop += 1

    print ("loop done : %d loops" % max_loop)
    print ("final rank_list : ", rank_list)
    return rank_list

def isPairExist(pair, d):
    return pair in d or (pair[1], pair[0]) in d

# convert 2D list to dictionary (key : ordered pair (i, j), value : similarity btw ith image & jth image)
# + dictionary will not store (i, i) -> 1
def dlist2dict(dlist):
    d = dict()
    for i in range(len(dlist)):
        for j in range(len(dlist[i])):
            if i == j:
                continue
            pair = (i, j)
            if not isPairExist(pair, d):
                d[pair] = dlist[i][j]
    return d

if __name__ == "__main__":

    sim_dlist = [
        [1, 0.8, 0.6, 0.3, 0.2],
        [0.8, 1, 0.55, 0.25, 0.18],
        [0.6, 0.55, 1, 0.58, 0.62],
        [0.3, 0.25, 0.58, 1, 0.9],
        [0.2, 0.18, 0.62, 0.9, 1]
    ]

    sim_permu = dlist2dict(sim_dlist)
    '''
    sim_permu = {
        (0,1) : 0.8,
        (0,2) : 0.6,
        (0,3) : 0.3,
        (0,4) : 0.2,
        (1,2) : 0.55,
        (1,3) : 0.25,
        (1,4) : 0.18,
        (2,3) : 0.58,
        (2,4) : 0.62,
        (3,4) : 0.9
    }
    '''

    get_ranks_by_dict([1] * 5, sim_permu, threshold=0.3, df=0.85, max_loop=100)