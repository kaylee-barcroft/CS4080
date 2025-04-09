# Exercise Set D
import numpy as np
import matplotlib.pyplot as plt
M = 7 # len(A)
ABET = {'A','B','C','D','E','F','G'} # A
KEY = 'C' # K

def main():
    dataset = []
    dataset_sizes = [5,10,20,50]
    
    for n in dataset_sizes:
        dataset.append(np.random.choice(ABET, n))
    
        counts = {i:0 for i in range(-1,n)}
        counts[n] = search(dataset[n], n, KEY)


def search(A, n, K):
    i = 0
    for i in range(n):
        if A[i] == K:
            return i
    return -1
