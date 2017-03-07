import os, sys
import math

def process_log(filepath, start_iter=0,  start_epoch = 0):
    data = open(filepath, 'r').read().strip().split('\n')

    start_ind = [x for x in range(len(data)) if 'Iteration' in data[x]][0]# and if 'Epoch: ' + str(start_epoch)][0]
    #start_ind = [x for x in range(len(data)) if 'Iteration: '+str(start_iter) in data[x] and if 'Epoch: '+str(start_epoch)][0]
    end_ind = [x for x in range(len(data)) if 'Iteration:' in data[x]][-1]
    data = data[start_ind:end_ind+2]
    data = [x for x in data if not 'Iteration' in x and not 'Epoch' in x]
    print(len(data))
    print(data[:10])
    return data

process_log('/raid/dkj755/rnn_alldata_rnn1024_logs.txt')
