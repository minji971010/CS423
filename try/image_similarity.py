import numpy as np
from sklearn import decomposition
from sklearn import datasets
from skimage import measure
import cv2
import sys
import os
import pyprind

def sigmoid(x):
    return 1/(1+np.exp(-x))

def PCA(X):
    """
    X : ndarray. data for PCA
    return : ndarray. PCAed datas
    """
    pca = decomposition.PCA(n_components=1)
    pca.fit(X)
    X = pca.transform(X)
    return X

def match(des1,des2,bf):
    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # BFMatcher with default params
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des1, des2, k=2)

#     # Apply ratio test
#     good = []
#     for m,n in matches:
#         if m.distance < 0.9 * n.distance:
#             good.append([m])
    good = [x for x in matches if x.distance < 50]
    return good

def knn_match(img1,img2,kp1,kp2,des1,des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.9 * n.distance:
            good.append([m])

    # Draw first 10 matches.
    knn_image = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    return good

def ORB(img1,img2):
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None) 
    kp2, des2 = orb.detectAndCompute(img2, None)
    if(des1 is None or des2 is None):
        return 0,0
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    good1 = match(des1,des2,bf)
    good2 = knn_match(img1,img2,kp1,kp2,des1,des2)
    return len(good1),len(good2)


def SIFT(img1,img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None) 
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    good1 = match(des1,des2,bf)
    good2 = knn_match(img1,img2,kp1,kp2,des1,des2)
    return len(good1),len(good2)

def SURF(img1,img2):
    surf = cv2.xfeatures2d.SURF_create()
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    good1 = match(des1,des2,bf)
    good2 = knn_match(img1,img2,kp1,kp2,des1,des2)
    return len(good1),len(good2)

def BRISK(img1,img2):
    brisk = cv2.BRISK_create()

    kp1, des1 = brisk.detectAndCompute(img1, None) 
    kp2, des2 = brisk.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=Falses)
    good1 = match(des1,des2,bf)
    good2 = knn_match(img1,img2,kp1,kp2,des1,des2)
    return len(good1),len(good2)

def SIMM(img1,img2):
    return 1/(1e-6+measure.compare_ssim(img1, img2,multichannel=True))

def compare_hist(img1,img2):
    assert img1.shape == img2.shape, "Two image shape is difference"
    
    temp1 = cv2.calcHist([img1],[0,1],None,[180,256],[0,180,0,256])
    temp2 = cv2.calcHist([img2],[0,1],None,[180,256],[0,180,0,256])
    err = cv2.compareHist(temp1, temp2, cv2.HISTCMP_CHISQR)/(img1.shape[0]*img1.shape[1])
    return 1/(err+1e-6)

def normalized_cross_correlation_ratio(img1,img2):
    assert img1.shape == img2.shape,"Two image shape is difference"
    
    w,h,c = img1.shape
    X = np.sum(np.square(img1))
    Y = np.sum(np.square(img2))
    Z = np.sum(np.multiply(np.abs(img1),np.abs(img2)))
    return Z/np.sqrt(X*Y)

def get_all_feat(img1,img2):
    """
    img1,img2 : ndarray. img which extract feature from.
                         img size should be the same
    return : ndarray. a feature of img
    """
    assert img1.shape == img2.shape, "Two image shape is difference"
    
#     feat_list = [ORB,SIFT,SURF,BRISK]
    feat_list = [ORB]
#     feat_list2 = [SIMM,compare_hist,normalized_cross_correlation_ratio]
    feat_list2 = [compare_hist,SIMM]
#     feat_list2 = []
    z = []
    for func in feat_list:
        a,b = func(img1,img2)
#         z.append(a)
        z.append(b)
    for func in feat_list2:
        z.append(func(img1,img2))
    return np.array(z,dtype='float32')

def feature_extraction(imgs):
    """
    imgs : list of (img1,img2)
           img size should be the same
           imgs should be scaled to [0,1]
    return : ndarray of PCAed feature
    """
    x = []
    bar = pyprind.ProgBar(len(imgs), stream=sys.stdout)
    for img1,img2 in imgs:
        x.append(get_all_feat(img1,img2))
        bar.update()
    X = np.array(x,dtype='float32')
#     X = PCA(X)
# PCA 잘 안돼서 걍 안씀
    X = (X-np.mean(X,axis=0))/np.std(X,axis=0)
    X = sigmoid(X)
    return X

def main():
    img_path = sys.argv[1]
    shape = (600,400)
    img_files = os.listdir(img_path)
    img_files.sort()
    img_files = [file for file in img_files if file.endswith('g')]
    print(img_files)
    imgs = [cv2.imread(os.path.join(img_path,img), cv2.IMREAD_COLOR) for img in img_files]
    cnt = 1
    for img in imgs:
        if(img is None):
            print(cnt)
        cnt+=1
    imgs = [cv2.resize(img, dsize=shape, interpolation=cv2.INTER_AREA) for img in imgs]
    pair = []

    for i in range(len(imgs)):
        for j in range(len(imgs)):
            if(i==j):
                continue
            else:
                pair.append((imgs[i],imgs[j]))
    features = feature_extraction(pair)
    np.save(img_path,features)
    print(features)
if __name__ == '__main__':
    main()