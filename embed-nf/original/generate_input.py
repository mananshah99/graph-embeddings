# generate input from rewired graphs
# ex: /dfs/scratch0/manans/rewire-exp-er

import sys
import os

directory = sys.argv[1]
output = open('rewire.csv', 'wb+')

files = os.listdir(directory)
files = sorted(files, key = lambda x : int(x.split('.')[0]))

total_rewires = len(files)

output.write('graph,percent-rewire\n')
for f in files:
    output.write(directory + f + ',' + str(float(f.split('.')[0])/total_rewires) + '\n')

output.close()
