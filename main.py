from _pagerank import pagerank
import operator
import numpy as np

def data(): # Group A, B, C size is same. Case 1.
    # make the data randomly
    l = np.zeros(shape=(30,30))

    for i in range(30):
        for j in range(30):
            if(i<10):
                if(j<10):
                    l[i,j] = np.random.uniform(0.9,1)
                elif(10<=j<20):
                    l[i,j] = np.random.uniform(0,0.3)
                elif(20<=j<30):
                    l[i,j] = np.random.uniform(0.3,0.4)
            elif(10<=i<20):
                if(j<10):
                    l[i,j] = np.random.uniform(0,0.3)
                elif(10<=j<20):
                    l[i,j] = np.random.uniform(0.9,1)
                elif(20<=j<30):
                    l[i,j] = np.random.uniform(0.3,0.4)
            elif(20<=i<30):
                if(j<10):
                    l[i,j] = np.random.uniform(0.3,0.4)
                elif(10<=j<20):
                    l[i,j] = np.random.uniform(0.3,0.4)
                elif(20<=j<30):
                    l[i,j] = np.random.uniform(0.7,0.9)

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

def new_data(): # Group C size is large. Case 2.
    # make the data randomly
    l = np.zeros(shape=(30,30))

    for i in range(30):
        for j in range(30):
            if(i<5):
                if(j<5):
                    l[i,j] = np.random.uniform(0.9,1)
                elif(5<=j<10):
                    l[i,j] = np.random.uniform(0,0.3)
                elif(10<=j<30):
                    l[i,j] = np.random.uniform(0.3,0.4)
            elif(5<=i<10):
                if(j<5):
                    l[i,j] = np.random.uniform(0,0.3)
                elif(5<=j<10):
                    l[i,j] = np.random.uniform(0.9,1)
                elif(10<=j<30):
                    l[i,j] = np.random.uniform(0.3,0.4)
            elif(10<=i<30):
                if(j<5):
                    l[i,j] = np.random.uniform(0.3,0.4)
                elif(5<=j<10):
                    l[i,j] = np.random.uniform(0.3,0.4)
                elif(10<=j<30):
                    l[i,j] = np.random.uniform(0.7,0.9)

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
def nnew_data(): # Group C size is small. Case 3.
    # make the data randomly
    l = np.zeros(shape=(30,30))

    for i in range(30):
        for j in range(30):
            if(i<13):
                if(j<13):
                    l[i,j] = np.random.uniform(0.9,1)
                elif(13<=j<26):
                    l[i,j] = np.random.uniform(0,0.3)
                elif(26<=j<30):
                    l[i,j] = np.random.uniform(0.6,0.8)
            elif(13<=i<26):
                if(j<13):
                    l[i,j] = np.random.uniform(0,0.3)
                elif(13<=j<26):
                    l[i,j] = np.random.uniform(0.9,1)
                elif(26<=j<30):
                    l[i,j] = np.random.uniform(0.6,0.8)
            elif(26<=i<30):
                if(j<13):
                    l[i,j] = np.random.uniform(0.6,0.8)
                elif(13<=j<26):
                    l[i,j] = np.random.uniform(0.6,0.8)
                elif(26<=j<30):
                    l[i,j] = np.random.uniform(0.7,0.9)

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
  G = data() # Case 1. size A = size B = size C
  # G = new_data() # Case 2. size C >> size A = size B
  # G = nnew_data() # Case 3. Size C << size A = size B
  # df = 0.15
  output = pagerank(G) # , df=df)
  sorted_output = sorted(output.items(), key=operator.itemgetter(1))
  reverse_sorted = sorted_output.reverse()
  print("-------------------------- Case 1 ---------------------------")
  print ("This is the sorted output:", sorted_output)
  print ("Top 10 values are", sorted_output[:9])
  print ("---------------------------------------------------------------")

main()
