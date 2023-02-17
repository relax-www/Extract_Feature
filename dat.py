import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.stats
from sklearn.model_selection import KFold
import pandas as pd
import pickle

def read_dat(PATH):
    f=open(PATH,encoding='utf-8')
    sentimentlist = []
    for line in f:
        s=line.strip()
        s = s.split("  ")
        s = [float(z) for z in s]
        sentimentlist.append(s)
    f.close()
    read_data=np.array(sentimentlist)#480*52
    if read_data.shape[0]<read_data.shape[1]:
        read_data=read_data.T
    return read_data
