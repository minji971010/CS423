import numpy as np
import cv2
import itertools

def get_similarity(path1, path2):
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    res = None

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    total_matchp_count = len(des1) if len(des1) > len(des2) else len(des2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches) / total_matchp_count

def get_similarity_dict(paths):
    res = dict()
    cnt = 0
    for i, j in itertools.combinations(range(len(paths)), 2):
        cnt += 1
        if cnt % 20 == 0:
            print ("Getting similarity... trial %d" % cnt)
        res[(i, j)] = get_similarity(paths[i], paths[j])
    return res
        


    
    