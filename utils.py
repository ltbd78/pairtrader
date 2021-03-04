import numpy as np

def rollingMA(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    ma = (cumsum[n:] - cumsum[:-n]) / n
    ma_padded = np.insert(ma, 0, np.full(n-1, np.nan))
    return ma_padded

def rollingSD(x, n):
    sd = np.full(len(x), np.nan)
    for i in range(n, len(x)+1):
        window = x[i-n:i]
        sd[i-1] = np.std(window, ddof=1)
    return sd