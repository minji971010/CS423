from _pagerank import pagerank
import operator
import numpy as np

def data():
    # make the data randomly
    l = np.zeros(shape=(30,30))

    for i in range(30):
        for j in range(30):
            if(i<10):
                if(j<10):
                    l[i,j] = np.random.uniform(0.8,1)
                elif(10<=j<20):
                    l[i,j] = np.random.uniform(0,0.4)
                elif(20<=j<30):
                    l[i,j] = np.random.uniform(0.5,0.9)
            elif(10<=i<20):
                if(j<10):
                    l[i,j] = np.random.uniform(0,0.4)
                elif(10<=j<20):
                    l[i,j] = np.random.uniform(0.8,1)
                elif(20<=j<30):
                    l[i,j] = np.random.uniform(0.5,0.9)
            elif(20<=i<30):
                if(j<10):
                    l[i,j] = np.random.uniform(0.5,0.9)
                elif(10<=j<20):
                    l[i,j] = np.random.uniform(0.5,0.9)
                elif(20<=j<30):
                    l[i,j] = np.random.uniform(0.8,0.9)

    # make the matrix symmetric 
    for i in range(30):
        for j in range(30):
            if (i<j):
                l[i,j] = l[j,i]

    # make the data to dict
    # G[u][v] = weight
    G = dict()
    for i in range(30):
        tmp = dict()
        s =0
        for j in range(30):
            tmp=dict()
        for j in range(30):
            if(i==j):
                continue
            else:
                tmp[j] = l[i,j]
        G[i] = tmp

    return G

def main():
  G = data()
  output = pagerank(G)
  sorted_output = sorted(output.items(), key=operator.itemgetter(1))
  reverse_sorted = sorted_output.reverse()
  print ("This is the sorted output:", sorted_output)
  print ("Top 10 values are", sorted_output[:9])

main()
